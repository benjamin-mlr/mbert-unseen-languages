from env.importing import *
from io_.info_print import printing
from io_.dat.constants import UNK_ID


def construct_word_embedding_table(word_dim, word_dictionary, word_embed_init_toke2vec, verbose=1):
    scale = np.sqrt(5.0 / word_dim*10)
    # +1 required for default value
    table = np.zeros([len(word_dictionary) + 1, word_dim], dtype=np.float32)
    # WARNING: it means that unfilled commodities will get 0 which is the defulat index !!
    if verbose >= 1:
        print("Initializing table with shape {} based onword_dictionary and word_dim  ".format(table.shape))
    table[UNK_ID, :] = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
    oov = 0
    inv = 0
    var = 0
    mean = 0
    var_oov = 0
    mean_oov = 0
    for word, index in word_dictionary.items():

        if word in word_embed_init_toke2vec:
            embedding = word_embed_init_toke2vec[word]
            inv += 1
            #print("PRETRAINED VECTOR", index, word, embedding)
            mean += np.mean(embedding)
            var += np.var(embedding)
        elif word.lower() in word_embed_init_toke2vec:
            embedding = word_embed_init_toke2vec[word.lower()]
            #print("LOWER PRETRAINED VECTOR", index, word, embedding)
            inv += 1
            mean += np.mean(embedding)
            var += np.var(embedding)
        else:
            if word == "the":
                print("word ", word, " --> not accounted")
            embedding = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
            mean_oov += np.mean(embedding)
            var_oov += np.var(embedding)
            #print("RANDOMY GENERATED", index, word, embedding)
            oov += 1
        table[index, :] = embedding
        #print("repeat", table[index, :])
    printing("W2V INFO : Mean of preloaded w2v {} var {} "
             "while the one generated randomly have {} mean and {} var in average",
             var=[mean/inv, var/inv, mean_oov/oov, var_oov/oov ], verbose_level=1, verbose=verbose)
    printing('W2V INFO  : OOV: %d/%d (%f rate (percent)) in %d' % (oov, len(word_dictionary) + 1, 100 * float(oov / (len(word_dictionary) + 1)), inv), verbose_level=1, verbose=verbose)
    word = "the"
    print("word {} of index {} has vector {} ".format(word, index, embedding))
    return torch.from_numpy(table)