import argparse

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="./data", type=str)
    parser.add_argument("--output", default="./output_data", type=str)
    # store true
    parser.add_argument("--remove_zip", action="store_true")
    parser.add_argument("--remove_unused_dcm", action="store_true")
    parser.add_argument("--tarball_name", default="output", type=str)
    args = parser.parse_args()
    print("Processing DEXA data UKB")
    utils.process_dexa_data_ukb(
        args.dataset_path,
        args.output,
        remove_zip=args.remove_zip,
        remove_unused_dcm=args.remove_unused_dcm,
    )
    print("Processing completed")
    print("Creating tarball")
    utils.make_tarball(args.output, args.output, args.tarball_name)
    print("Tarball created")
