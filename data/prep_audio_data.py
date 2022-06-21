import sox
import os
import argparse
from multiprocessing import Pool

def resample_file(file, in_dir, out_dir, tfm):
    out_path = os.path.join(out_dir, file)
    in_path = os.path.join(in_dir, file)
    if os.path.isfile(out_path):
        print(f'{out_path} already exists.')
    elif not file.lower().endswith('.wav'):
        print(f'{in_path}: invalid file type.')
    else:
        success = tfm.build_file(input_filepath=in_path, output_filepath=out_path)
        if success:
            print(f'Succesfully saved {in_path} to {out_path}')


def resample_data(args):
    in_dir = args.data_dir
    out_dir = args.out_dir
    target_sr= args.target_sr
    n_jobs = args.n_jobs

    tfm = sox.Transformer()
    tfm.set_output_format(rate=target_sr)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    files = [file for file in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, file))]
    with Pool() as p:
        p.starmap(resample_file, [(file, in_dir, out_dir, tfm) for file in files], chunksize=len(files)//n_jobs)



def parse_args():
    parser = argparse.ArgumentParser(description='Resample data.')
    parser.add_argument('--data_dir', help='directory containing source files')
    parser.add_argument('--out_dir', help='directory to write target files')
    parser.add_argument('--target_sr', type=int, help='target sample rate')
    parser.add_argument('--n_jobs', type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)

    resample_data(args)
    print(f'Done resampling to target rate {args.target_sr}.')


if __name__ == '__main__':
    main()