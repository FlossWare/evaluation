"""Measure lexical-v1 analyzer throughput and offset correctness on fixed fixtures."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import resolve_sut_root,summarize_ms,write_result
SUT_ROOT=resolve_sut_root()
sys.path.insert(0,str(SUT_ROOT/"services"/"tokenization"))
from tokenization_service import tokenize

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--iterations",type=int,default=100); p.add_argument("--output",default="benchmarks/results"); a=p.parse_args()
 if a.iterations<1: p.error("iterations must be positive")
 fixtures=[json.loads(x) for x in (Path(__file__).resolve().parents[1]/"fixtures"/"tokenization.jsonl").read_text().splitlines() if x.strip()]
 cases=[]
 for f in fixtures:
  fn=lambda: tokenize("lexical-v1",f["text"]); expected=fn(); samples=[]
  for _ in range(a.iterations):
   t=time.perf_counter(); tokens=fn(); samples.append((time.perf_counter()-t)*1000)
  folded="".join(ch.casefold() if len(ch.casefold())==1 else ch.lower() for ch in f["text"])
  valid=all(0<=x["start"]<x["end"]<=len(f["text"]) and folded[x["start"]:x["end"]]==x["term"] for x in tokens)
  cases.append({"case":f["case"],"input_chars":len(f["text"]),"token_count":len(tokens),"timing":summarize_ms(samples),
   "chars_per_second":len(f["text"])/(sum(samples)/1000/len(samples)) if samples else None,"deterministic":tokens==expected,"offsets_valid":valid})
 out=write_result("tokenization",{"profile":"lexical-v1","iterations":a.iterations,"cases":cases},a.output); print(out)
if __name__=="__main__": main()
