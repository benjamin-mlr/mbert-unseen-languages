
import argparse

def append_hash(dir_source, target):
	with open(dir_source,"r") as f :
		with open(target,"w") as g:
			for line in f :
				if line.startswith("1\t"):
					g.write("#\n")
				g.write(line)

if __name__=="__main__" :
  args_parser = argparse.ArgumentParser(description="ELMoLex Parser - Testing")
  args_parser.add_argument('--dir', required=True, type=str)
  args_parser.add_argument('--target', type=str,required=True)
  
  args =  args_parser.parse_args()
  append_hash(args.dir,args.target)