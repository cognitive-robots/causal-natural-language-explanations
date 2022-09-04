# |**********************************************************************;
# Project           : Why Stop Now? Causal Natural Language Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

import argparse
import Step4_preprocessing_explanation as Step4
import Step5_train_Generator as Step5_train
import Step5_1_test_Generator as Step5_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DESC')
    parser.add_argument('--pre_process', type=int, default=1, help='Include download and pre-process? then 1 else 0.')
    parser.add_argument('--train', type=int, default="0", help='1: train, 2: test')
    args = parser.parse_args()

    # Run 1x pre-processing for data
    if args.pre_process == 1:
        Step4.main()

    # Train the LSTM Language Generator
    if args.train == 1:
        Step5_train.main()

    # Test the LSTM Language Generator and generate output csv
    if args.train == 2:
        Step5_test.main(["--extractText=True"])

