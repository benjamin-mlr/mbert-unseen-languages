


# 1 : For gold file : Read row by row and only select conlly rows other copy paste it 
# 1b : For pred file :  Read row by row and only select conlly rows other copy paste it + Assert alignement with gold at the token level 
# 2 : copy/paste from gold all information except upos, xpos and feats 
# 2b : copy/paste from pred upos, xpos and feats 


def write_buffer(self):

    for seq_no in range(len(self.__out_data)):
      sent_tokens, raw_lines = self.__out_data[seq_no]
      cur_ti = 0
      for ud_tokens in raw_lines:
        idi, form, lemma, upos, xpos, feats, head, deprel, deps, misc = ud_tokens
        if '-' not in idi and '.' not in idi:
          cur_model_tokens = sent_tokens[cur_ti]
          #head, typ = str(cur_model_tokens[1]), cur_model_tokens[2]
          upos = str(cur_model_tokens[1])
          cur_ti+=1

        self.__source_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(idi, form, lemma, upos, xpos, feats, head, deprel, deps, misc))
      self.__source_file.write("\n")
