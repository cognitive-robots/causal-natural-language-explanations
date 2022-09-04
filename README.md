# Why Stop Now? Causal Natural Language Explanations for Automated Commentary Driving

This repository contains code for the following paper (currently under review):

Marc Alexander Kühn, Daniel Omeiza and Lars Kunze, Why Stop Now? Causal Natural Language Explanations for Automated Commentary Driving. ICRA 2023.

### Instructions
- Build and run the Dockerfile to have all necessary dependencies
- Follow the steps according to pipeline_controller.py to train and test the feature encoder and vehicle controller
- Follow the steps according to pipeline_generator.py to train and test the language generator
- For each new dataset / model combination, adjust the src/config-files accordingly (used hyperparameters can be found in the paper)
- Follow the code adjustments for specific datasets and models stated below
- BDD-X dataset: https://github.com/JinkyuKimUCB/BDD-X-dataset
- SAX Explanation Driving Dataset will be released later during the year

For SAX dataset:
- Use Step1_SAX_preproccesing.py in step 1
- dataloader_CNN_course2.py and dataloader_VA_course2: Change iterator 1 (BDD-X) to 10 (SAX) in lines 43-47. Change multiplicator from 1 (BDD-X) to 10 (SAX) in line 61.
- Step2_train_CNNonly.py: Change import dataloader_CNN (BDD-X) to import dataloader_CNN_course2 (SAX) in line 18
- Step3_train_Attention.py: Change import dataloader_VA (BDD-X) to import dataloader_VA_course2 (SAX) in line 20
- Step4_preprocessing_explanation.py: Change build_feat_matrix (BDD-X) to build_feat_matrix_SAX (SAX) in line 78
- Step5_1_test_Generator.py: Change test set size in line 90.

For PoS prediction:
- Step4_preprocessing_explanation.py: Change process_caption_data to process_caption_data_w_pos in line 53. Change build_caption_vector to build_caption_vector_w_pos in line 75.

For different language generator model:
- Step5_1_test_Generator.py: Change src.LSTM_Gen_slim_v2 to src.LSTM_Gen_slim_v2_pos_penalties (PoS + Penalties) or src.LSTM_Gen_slim_v2_pos (PoS) or src.LSTM_Gen_slim_v2_w_penalties (Token Penalties) in line 22.
- Step5_train_Generator.py: Change src.LSTM_Gen_slim_v2 to src.LSTM_Gen_slim_v2_pos_penalties (PoS + Penalties) or src.LSTM_Gen_slim_v2_pos (PoS) or src.LSTM_Gen_slim_v2_w_penalties (Token Penalties) in line 17.

### References
This code is based on the publication and code by Kim et al. [1]

[1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
