import glob
import util

TRAINING_LABEL_PATHS = glob.glob('{}/sign_labels/train/*.xml'.format(util.get_resources_directory()))

exit_sign_count = 0
bathroom_sign_count = 0

# loop over each file
for training_label_path in TRAINING_LABEL_PATHS:
    training_label = open(training_label_path, 'r')

    # loop over each line in the file
    while True:
        training_line = training_label.readline()
        if training_line == '':
            break
        elif 'exit sign' in training_line:
            exit_sign_count += 1
        elif 'bathroom sign' in training_line:
            bathroom_sign_count += 1

print('exit sign count: {}'.format(exit_sign_count))
print('bathroom sign count: {}'.format(bathroom_sign_count))
