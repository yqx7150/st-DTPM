from metrics import fid_score_
import argparse
import torch

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    fid = fid_score_( real_image_folder=f"{args.sample_name}/real",
                    generated_image_folder=f"{args.sample_name}/gen",
                    device=device)
    print(fid)

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_name", type=str, default="../assert/sample_AD_EC")
    args = parser.parse_args()

    main(args=args)