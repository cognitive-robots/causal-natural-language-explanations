# |**********************************************************************;
# Project           : Textual Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

import argparse
import Step1_preprocessing as Step1
import Step2_train_CNNonly as Step2_train
import Step2_1_test_CNNonly as Step2_test
import Step3_train_Attention as Step3_train
import Step3_1_test_Attention as Step3_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DESC')
    parser.add_argument('--pre_process', type=int, default=1, help='Include download and pre-process? then 1 else 0.')
    parser.add_argument('--train', type=str, default="None", help='Train the networks?')
    parser.add_argument('--feat', type=int, default="0", help='1: train, 2: val, 3: test')
    parser.add_argument('--attn', type=int, default="0", help='1: train, 2: val')
    args = parser.parse_args()

    # Run 1x pre-processing for data
    if args.pre_process == 1:
        Step1.main()

    # Train the CNN Feature Extractor
    if args.train == "CNN":
        Step2_train.main()

    # After Training CNN, generate both feature sets
    if args.feat == 1:
        Step2_test.main(["--extractFeature=True"])
    if args.feat == 2:
        Step2_test.main(["--extractFeature=True", "--validation=True"])
    if args.feat == 3:
        Step2_test.main(["--extractFeature=True", "--test=True"])

    # Train VA LSTM Controller
    if args.train == "VA":
        Step3_train.main()

    # After VA Training, Extract Attention
    if args.attn == 1:
        Step3_test.main(["--extractAttn=True", "--extractAttnMaps=True"])
    if args.attn == 2:
        Step3_test.main(["--extractAttn=True", "--extractAttnMaps=True", "--validation=True"])
    if args.attn == 3:
        Step3_test.main(["--extractAttn=True", "--extractAttnMaps=True", "--test=True"])
