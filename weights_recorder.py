import os
import json
import sqlite3

from numpy import ndarray, asfarray, zeros, argmax, asarray, mean

from FFNN import FFNN


class WeightsRecorder:
    def __init__(self, data_delimiter=100000):
        # data_delimiter: sets due to dataset size, for example is dataset size is 110000,
        #               data_delimiter could be 100000, to get training_data set = 100000 and testing_data set = 10000
        self.dataset = self.init_dataset()
        self.training_data = self.dataset[:data_delimiter]
        self.testing_data = self.dataset[data_delimiter:]

        self.avg_success = 0
        self.successes = []
        self.highest_success = 0
        self.successful_weights = []

    def init_dataset(self):
        """Fill in the dataset with data about sport matches from files"""
        project_dir = os.path.dirname(__file__)
        files = os.listdir(project_dir + '/databases/')
        data = []
        for file in files:
            if '2020' in file:
                data += self.get_data(project_dir + '\\databases\\' + file)
        return data

    def train_net(self, nn, epochs):
        for _ in range(epochs):
            for record in self.training_data:
                # Marking each record with "True" or "False" label
                e1 = record[:26]
                e1.append(record[27])
                e1.append(record[29])
                inputs_set = asfarray(e1)
                targets_set = zeros(len(nn.weights[-1])) + 0.0001
                if record[26] > record[28]:
                    targets_set[0] = 0.9999
                else:
                    targets_set[1] = 0.9999
                nn.train_net(inputs_set, targets_set)

    def test_net(self, nn):
        scorecard = []

        for record in self.testing_data:
            e1 = record[:26]
            e1.append(record[27])
            e1.append(record[29])
            inputs_set = asfarray(e1)
            if record[26] > record[28]:
                correct_label = 0
            else:
                correct_label = 1
            output_list = nn.ask_net(inputs_set)
            label = argmax(output_list)
            if label == correct_label:
                scorecard.append(1)
            else:
                scorecard.append(0)

        success = mean(asarray(scorecard))
        print("Success = " + str(success)[:6])

        return success

    def set_net_highest_success(self, weights, success):
        """Set current weights as the most successful
            if they have better predicting accuracy if previous ones"""
        if self.highest_success < success:
            self.highest_success = success
            self.successful_weights = weights

        self.avg_success += success
        self.successes.append(success)

    def record_current_weights(self, weights, path=(os.path.dirname(__file__) + '/weights/***.json')):
        with open(path, 'w') as f:
            weights_dict = {"w"+str(number): list() for number in range(len(weights))}

            for number, w in enumerate(self.successful_weights):
                for i in w:
                    local_dict = {}
                    for counter, i_ in enumerate(i, 1):
                        local_dict[str(counter)] = i_
                    weights_dict["w"+str(number)].append(local_dict)

            json.dump(weights_dict, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    @staticmethod
    def get_data(path):
        data = []
        with sqlite3.connect(path) as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT * FROM matches''')
            query_data = cursor.fetchall()
            for row in query_data:
                # Pre-normalizing of parameters
                data_set = row[2:15] + row[16:29] + row[30:]
                data_set = list(data_set)
                data_set[0] = data_set[0] * 0.01
                data_set[13] = data_set[13] * 0.01
                data_set[26] = data_set[26] * 0.01
                data_set[26] = float(str(data_set[26])[:5])
                data_set[27] = float(str(data_set[27])[:6])
                data_set[28] = data_set[28] * 0.01
                data_set[28] = float(str(data_set[28])[:5])
                data_set[29] = float(str(data_set[29])[:6])
                data.append(data_set)
        return data


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():
    recorder = WeightsRecorder()
    # To check stability of net at a distance, init bigger a "times" variable
    times = 1
    nn_instance = None

    for _ in range(times):
        nn_instance = FFNN((28, 100, 100, 100, 100, 100, 2), 0.999999)
        epochs_amount = 10

        # Configure weights by training the n_net
        recorder.train_net(nn_instance, epochs_amount)

        # Test weights by testing the n_net with new data
        predicting_success = recorder.test_net(nn_instance)

        # Save weights if they are better then previous ones
        recorder.set_net_highest_success(nn_instance.weights, predicting_success)

        print("Success = " + str(predicting_success)[:6])

    # Save weights in .json file if they are satisfactory
    print('Avg_success = ' + str(recorder.avg_success / times))
    print('Highest_success = ' + str(recorder.highest_success))
    print('Do you want to rewrite weights of net?(y/n)')
    answer = input()
    if answer == 'y':
        file_path = os.path.dirname(__file__) + '/weights/***.json'
        recorder.record_current_weights(nn_instance.weights, file_path)


if __name__ == '__main__':
    main()
