# Makefile to assist generating files necessary at run-time.
# This does model training, for instance

# Scripts
ENCODE_SCRIPT = ksDatasetEncode.py
TRAIN_SCRIPT = trainModel.py
# this is only used at run-time by the server-side code.
# RUNTIME_SCRIPT = kickstarter.py

# Inputs
DATASET = ksDataset.csv
FAKE_USER_INPUT = successful.csv

# Outputs
TRAIN_INPUTS = encodeOut1.out encodeOut2.out
ENCODE_OUTPUTS = $(TRAIN_INPUTS) encodeOut3.out encodeOut4.out encodeOut5.out
MODEL_OUTPUTS = modelOut1.out modelOut2.out modelOut3.out modelOut4.out modelOut5.out modelOut6.out modelOut7.out modelOut8.out modelOut9.out modelOut10.out
ALL_OUTPUTS = $(ENCODE_OUTPUTS) $(MODEL_OUTPUTS)

.PHONY: all
all: $(ALL_OUTPUTS)

$(ENCODE_OUTPUTS): $(ENCODE_SCRIPT)

$(ENCODE_SCRIPT): $(DATASET) $(FAKE_USER_INPUT)
	python3 $@ $+ $(ENCODE_OUTPUTS)

$(MODEL_OUTPUTS): $(TRAIN_SCRIPT)

$(TRAIN_SCRIPT): $(TRAIN_INPUTS)
	python3 $@ $+ $(MODEL_OUTPUTS)

$(DATASET): check
	@echo "Checked $@"

$(FAKEUSER_INPUT): check
	@echo "Checked $@"


check:
