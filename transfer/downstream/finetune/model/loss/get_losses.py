

from transfer.downstream.finetune.env.imports import nn, torch, np, re, OrderedDict, pdb, json, os, shutil, tempfile, sys, boto3, requests, tqdm, ClientError, urlparse, logging



try:
    from pathlib import Path
    PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                   Path.home() / '.pytorch_pretrained_bert'))
except AttributeError:
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                              os.path.join(os.path.expanduser("~"), '.pytorch_pretrained_bert'))

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from transfer.downstream.finetune.model.settings import AVAILALE_PENALIZATION_MODE
from transfer.downstream.finetune.args.args_parse import parse_argument_dictionary


def _get_key_ponderation(name_layer, dict_per_layer):

    for prefix_reg in dict_per_layer:
        if re.match(prefix_reg, name_layer) is not None:
            return prefix_reg
        #if name_layer.startswith(prefix):
        #    return prefix
    raise(Exception("No prefix of {} found for layer {}  ".format(dict_per_layer, name_layer)))


def _get_n_param(param):
    n_param = 1
    for dim in param.size():
        n_param *= dim
    return n_param


def get_penalization(model_parameters, model_parameters_0, norm_order_per_layer, ponderation_per_layer, penalization_mode=None, pruning_mask=None):
    penalization_dic = OrderedDict()
    assert isinstance(ponderation_per_layer, dict), "{} should be dict ".format(ponderation_per_layer)
    #assert set(ponderation_per_layer) == set(norm_order_per_layer), "ERROR {} not same keys as {}".format(norm_order_per_layer, ponderation_per_layer)
    if penalization_mode is None:
        penalization_mode = "pruning"
    assert penalization_mode in AVAILALE_PENALIZATION_MODE, "ERROR {} shoul be in {}".format(penalization_mode, AVAILALE_PENALIZATION_MODE)
    penalization = 0

    for (name, param), (name_0, param_0) in zip(model_parameters, model_parameters_0.items()):
        assert name == name_0, "ERROR {} <> {}".format(name, name_0)
        key_norm = _get_key_ponderation(name, norm_order_per_layer)
        key_pond = _get_key_ponderation(name, ponderation_per_layer)
        n_param_layer = _get_n_param(param)
        # Each single unit parameter count the same (modulo ponderation_per_layer-)
        #print("SANITY CHECKING debugging param {} has norm {} for dim {} ".format(name, torch.norm(param_0, p=norm_order_per_layer[key]), n_param_layer))
        power = norm_order_per_layer[key_norm] if norm_order_per_layer[key_norm] == 2 else 1

        # new --
        if penalization_mode == "pruning":
            assert pruning_mask is not None, "ERROR pruning_mask needed"
            # only norm 2 supported so far
            # ponderation applies on the non pruned mask
            pruning_mask_non = 1-pruning_mask[name_0]
            diff = (param - param_0).flatten()
            norm_2_other = torch.sum((pruning_mask_non * diff) ** 2)
            norm_2_on_mask_param = torch.sum((pruning_mask[name_0] * (diff)) ** 2)
            _penalization = ponderation_per_layer[key_pond] * norm_2_other + 10 * norm_2_on_mask_param
            penalization += _penalization
            penalization_dic[name] = (n_param_layer, ponderation_per_layer[key_pond], _penalization.detach().cpu(), norm_2_other.detach().cpu(), norm_2_on_mask_param.detach().cpu())
        elif penalization_mode == "layer_wise":
            penalization += ponderation_per_layer[key_pond]*torch.norm((param-param_0).flatten(), p=norm_order_per_layer[key_norm])**power
            penalization_dic[name] = (n_param_layer, ponderation_per_layer[key_pond], (torch.norm((param.detach()-param_0).flatten(), p=norm_order_per_layer[key_norm])).cpu()**power)
        else:
            raise(Exception("penalization_mode {} not supported".format(penalization_mode)))

    penalization_dic["all"] = ("all", "total", penalization)
    return penalization, penalization_dic


def get_loss_multitask(loss_dict, ponderation):

    loss = 0
    ponderation = parse_argument_dictionary(ponderation, loss_dict.keys())

    for label_loss in loss_dict:
        loss += ponderation[label_loss] * loss_dict[label_loss]
    return loss




def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + '.json'
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata['url']
    etag = metadata['etag']

    return url, etag

def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    #@wraps(func) CHANGE BY ME
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise

    return wrapper


@s3_request
def s3_etag(url):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url, cache_dir=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError("HEAD request failed for url {} with status code {}"
                          .format(url, response.status_code))
        etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w', encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path

def cached_path(url_or_filename, cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https', 's3'):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == '':
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

