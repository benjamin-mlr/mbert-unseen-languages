

from transfer.downstream.finetune.env.imports import os, re



# NB : lots of the datasets directory are in project_variables
# this files aims to group all of those at some point


DATA_UD = os.environ.get("DATA_UD", "/Users/bemuller/Documents/Work/INRIA/dev/parsing/data/Universal-Dependencies-2.4")
DATA_UD_25 = os.environ.get("DATA_UD_25", "/Users/bemuller/Documents/Work/INRIA/dev/parsing/data/Universal-Dependencies-2.5")
DATA_WIKI_NER = os.environ.get("DATA_WIKI_NER", "/Users/bemuller/Documents/Work/INRIA/dev/parsing/data/wikiner")
DATA_UD_RAW = os.environ.get("DATA_UD_RAW", "/Users/bemuller/Documents/Work/INRIA/enseignements/statapp/year2/lm/attention_data")

DATASET_CODE_LS = ['af_afribooms', 'ar_nyuad', 'ar_padt', 'be_hse', 'bg_btb', 'bxr_bdt', 'ca_ancora', 'cop_scriptorium',
                   'cs_cac', 'cs_cltt', 'cs_fictree', 'cs_pdt', 'cu_proiel', 'da_ddt', 'de_gsd', 'el_gdt', 'en_esl',
                   'en_ewt', 'en_gum', 'en_lines', 'en_partut', 'es_ancora', 'es_gsd', 'et_edt', 'et_ewt', 'eu_bdt',
                   'fa_seraji', 'fi_ftb', 'fi_tdt', 'fr_ftb', 'fr_gsd', 'fr_partut', 'fr_sequoia', 'fr_spoken', 'fro_srcmf',
                   'ga_idt', 'gl_ctg', 'gl_treegal', 'got_proiel', 'grc_perseus', 'grc_proiel', 'he_htb', 'hi_hdtb', 'hr_set',
                   'hsb_ufal', 'hu_szeged', 'hy_armtdp', 'id_gsd', 'it_isdt', 'it_partut', 'it_postwita', 'it_vit', 'ja_bccwj',
                   'ja_gsd', 'kk_ktb',  'ko_gsd', 'ko_kaist', 'la_ittb', 'la_perseus', 'la_proiel', 'lt_alksnis', 'lt_hse',
                   'lv_lvtb', 'lzh_kyoto', 'mr_ufal',  'nl_alpino', 'nl_lassysmall', 'no_bokmaal', 'no_nynorsk', 'no_nynorsklia',
                   'orv_torot', 'pl_lfg', 'pl_pdb', 'pt_bosque', 'pt_gsd', 'qhe_hiencs', 'ro_nonstandard', 'ro_rrt', 'ru_gsd', 'ru_syntagrus',
                   'ru_taiga', 'sk_snk', 'sl_ssj', 'sl_sst', 'sme_giella', 'sr_set', 'sv_lines', 'sv_talbanken', 'swl_sslc', 'ta_ttb',
                   'te_mtg', 'tr_imst',  'uk_iu', 'ur_udtb', 'vi_vtb', 'wo_wtb', 'zh_gsd',
                   "fr_gsd+fr_sequoia+fr_spoken+fr_partut",
                   "fr_partut_demo",
                   "alg_arabizi", "arabizi",
                   "fr_ugc_cb2", "en_ugc_owoputi",
                   "en_ugc_lexnorm", "en_tweet", "fr_ftb_ner_pos","fr_ftb_pos_ner",
                  "ar_padt_1500_fr_partut", "ar_padt_1500_it_partut", "ar_padt_1500_en_partut", "ar_padt_1500",
                   "fr_partut_it_partut", "fr_partut_en_partut","it_partut_en_partut", "alg_arabizi_lang",
                   "alg_arabizi_l_60q", "alg_arabizi_no_code_mixed", "alg_arabizi_b_10q", "alg_arabizi_15_30q", "alg_arabizi_10_15q","alg_arabizi_glose", "alg_arabizi_glose_alg_arabizi",
                   "ar_padt_1500_fr_partut","it_partut_mt_mudt", "ar_padt_1500_mt_mudt","alg_arabizi_mt_mudt","fr_partut_mt_mudt",
                   "alg_arabizi_mixed_t1", "alg_arabizi_mixed_t2", "alg_arabizi_mixed_t3", "alg_arabizi_arabizi","alg_arabizi_mt_mudt_fr_partut",

                   # 2.5
                   # only test
                   

                   # src : ru_gsd fr_partut, hu_szeged, english

                   # NEW SAMPLE
                   'it_partut_1500_1',

                   'en_partut_1500_1',

                   'fr_gsd_1500_1',
                   'fr_gsd_5000_1',

                   'ar_padt_1500_1',
                   'ar_padt_5000_1',

                   'ru_gsd_1500_1',

                   'tr_imst_1500_1',

                   'de_gsd_1500_1',
                   'de_gsd_5000_1',

                   # 2.5
                   'got_proiel_1500_1',

                   'fro_srcmf_1500_1',
                   'fro_srcmf_5000_1',

                   'orv_torot_1500_1',
                   'orv_torot_5000_1',

                   'ug_udt_1500_1',




                  "hr_set_1500_1",  "sr_set_1500_1" , "hi_hdtb_1500_1", "ur_udtb_1500_1" , 
                  "hr_set",  "sr_set" , "hi_hdtb", "ur_udtb" , "ta_ttb" ,
                "en_ewt_1500_1", 


                "en_over_lap_fr_partut",
                
                "id_gsd_1500_1",                
                'ug_udt_1123_1',
                "narabizi", "tr_imst_id_gsd_ar_padt_500_x3",
                "tr_imst_1000_ar_padt_500","ja_gsd_1500_1",
                'fr_wiki', 'ar_wiki', 'id_wiki', 'de_wiki', 'ru_wiki', 'tr_wiki', 'en_wiki',
                'fr_wiki_5000_1', 'ar_wiki_5000_1', 'id_wiki_5000_1', 'de_wiki_5000_1', 'ru_wiki_5000_1', 'tr_wiki_5000_1', 'en_wiki_5000_1',

                "en_conll03", 

                # OLD LANG
                "orv_torot", "fro_srcmf", "got_proiel", 

                # NEW LOW RESSOURCE 
                'ug_udt',"pcm_nsc","mt_mudt", "narabizi"

                # no dev , super small treain 
                "kmr_mg", # Kurmanji # 20 train sent
                "olo_kkpp", # LIVI  # 20 train sent

                # TEST ONLY                  
                 "mdf_jr", # MOSKA
                 "myv_jr", # erzya wals
                  "bm_crb", # bambara wals
                 "am_att",  # Amharic wals
                 "fo_oft", # Faroese
                 "tl_trg", # TAGALOG
                 "gsw_uzh",  # Swiss german 
                 "koi_uh", # Komi 
                 "tr_wiki", "ru_wiki", "en_wiki", "ar_wiki", #"fi_wiki", "de_wiki", # ADD Persian (for ku), "german faroese" 
                "am_wiki", "ug_wiki", "sd_wiki", "ceb_wiki", "ku_wiki", "tl_wiki", "mt_wiki", "fo_wiki", 

                "bxr_bdt_latin", "myv_jr_latin", "olo_kkpp_cyr", "wo_wtb_cyr",

                # controlled experiments 
                "ru_wiki_latin", "ar_wiki_translit",  "ja_wiki_latin",

                ]


DATASET_CODE_LS.extend(["alg_arabizi_{}_fr_partut".format(n_sent_arabizi) for n_sent_arabizi in [50, 100, 200, 400, 800, 1172]])
DATASET_CODE_LS.extend(["alg_arabizi_{}_fr_gsd".format(n_sent_arabizi) for n_sent_arabizi in [50, 100, 200, 400, 800, 1172]])
DATASET_CODE_LS.extend(["alg_arabizi_10_2_fr_partut", "alg_arabizi_10_1_fr_partut", "alg_arabizi_10_2_fr_gsd", "alg_arabizi_10_1_fr_gsd"])


def get_dir_data0(set, data_code, demo=False):

    assert set in ["train", "dev", "test"]
    assert data_code in DATASET_CODE_LS, "ERROR {}".format(data_code)
    demo_str = "-demo" if demo else ""

    file_dir = os.path.join(DATA_UD, "{}-ud-{}{}.conllu".format(data_code, set, demo_str))

    assert os.path.isfile(file_dir), "{} not found".format(file_dir)

    return file_dir


def get_dir_data(set, data_code, demo=False):

    assert set in ["train", "dev", "test"], "{} - {}".format(set, data_code)
    assert data_code in DATASET_CODE_LS, "ERROR {}".format(data_code)
    demo_str = "-demo" if demo else ""
    # WE ASSSUME DEV AND TEST CANNOT FINISH by INTERGER_INTERGER IF THEY DO --> fall back to data_code origin
    if set in ["dev", "test"]:
        matching = re.match("(.*)_([0-9]+)_([0-9]+)$",data_code)
        if matching is not None:
            data_code = matching.group(1)
            print("WARNING : changed data code with {}".format(data_code))
        else:
            pass#print("DATA_CODE no int found  {} ".format(data_code))
    file_dir = os.path.join(DATA_UD, "{}-ud-{}{}.conllu".format(data_code, set, demo_str))
    try:
        assert os.path.isfile(file_dir), "{} not found".format(file_dir)
    except:
        try:
            file_dir = os.path.join(DATA_UD_25, "{}-ud-{}{}.conllu".format(data_code, set, demo_str))
            assert os.path.isfile(file_dir), "{} not found".format(file_dir)
            print("WARNING : UD 25 USED ")
        except Exception as e:
            print("--> data ", e)
            demo_str = ""
            file_dir = os.path.join(DATA_WIKI_NER, "{}-{}{}.conll".format(data_code, set, demo_str))
            assert os.path.isfile(file_dir), "{} not found".format(file_dir)
            print("WARNING : WIKI NER USED")
    return file_dir


def get_code_data(dir):
    matching = re.match(".*\/([^\/]+).*.conllu", dir)
    if matching is not None:
        return matching.group(1)
    else:
        matching = re.match(".*\/([^\/]+).*.conll", dir)
        if matching is not None:
            return matching.group(1)
    return "training_set-not-found"

# 1 define list of dataset code and code 2 dir dictionary
# 2 : from grid_run : call dictionary and iterate on data set code

# DATA DICTIONARY

