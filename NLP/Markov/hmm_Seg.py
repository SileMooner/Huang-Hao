import pickle
from tqdm import tqdm
import numpy as np
import os

def create_label(word):
    length = len(word)
    if length == 1:
        return "S"
    return "B" + "M" * (length - 2) + "E"

def generate_state_file(input_file="all_train_text.txt", output_file="all_train_state.txt"):
    if os.path.exists(output_file):
        return
    
    with open(input_file, "r", encoding="utf-8") as f, \
         open(output_file, "w", encoding="utf-8") as fw:
        for line in tqdm(f, desc="Processing text to state"):
            if line.strip():
                states = []
                for word in line.split():
                    if word:
                        states.append(create_label(word))
                fw.write(" ".join(states) + "\n")

class HMM:
    def __init__(self, text_file="all_train_text.txt", state_file="all_train_state.txt"):
        self.texts = [line.strip() for line in open(text_file, "r", encoding="utf-8").read().splitlines()[:200]]
        self.states = [line.strip() for line in open(state_file, "r", encoding="utf-8").read().splitlines()[:200]]
        self.state_index = {"B": 0, "M": 1, "S": 2, "E": 3}
        self.init_matrix = np.zeros(len(self.state_index))
        self.transition_matrix = np.zeros((len(self.state_index), len(self.state_index)))
        self.emission_matrix = {state: {"total": 0} for state in "BMSE"}

    def train(self):
        if os.path.exists("three_matrix.pkl"):
            self.init_matrix, self.transition_matrix, self.emission_matrix = pickle.load(open("three_matrix.pkl", "rb"))
            return
        
        for text, state in tqdm(zip(self.texts, self.states), desc="Training HMM"):
            words = text.split()
            state_labels = state.split()
            self.init_matrix[self.state_index[state_labels[0]]] += 1
            self.update_matrices(words, state_labels)
        self.normalize_matrices()
        pickle.dump([self.init_matrix, self.transition_matrix, self.emission_matrix], open("three_matrix.pkl", "wb"))

    def update_matrices(self, words, states):
        for prev_state, next_state in zip(states, states[1:]):
            self.transition_matrix[self.state_index[prev_state], self.state_index[next_state]] += 1
        
        for word, state in zip("".join(words), "".join(states)):
            if word in self.emission_matrix[state]:
                self.emission_matrix[state][word] += 1
            else:
                self.emission_matrix[state][word] = 1
            self.emission_matrix[state]["total"] += 1

    def normalize_matrices(self):
        total_initials = np.sum(self.init_matrix)
        if total_initials > 0:
            self.init_matrix /= total_initials
        
        for i in range(len(self.state_index)):
            total_transitions = np.sum(self.transition_matrix[i, :])
            if total_transitions > 0:
                self.transition_matrix[i, :] /= total_transitions
        
        for state in self.emission_matrix:
            total = self.emission_matrix[state]["total"]
            if total > 0:
                for word in self.emission_matrix[state]:
                    if word != "total":
                        self.emission_matrix[state][word] /= total

def viterbi(text, hmm):
    states = list(hmm.state_index.keys())
    V = [{}]
    path = {}
    for state in states:
        V[0][state] = hmm.init_matrix[hmm.state_index[state]] * hmm.emission_matrix[state].get(text[0], 0)
        path[state] = [state]

    for i in range(1, len(text)):
        V.append({})
        new_path = {}
        for current_state in states:
            (prob, state) = max((V[i-1][prev_state] * hmm.transition_matrix[hmm.state_index[prev_state], hmm.state_index[current_state]] * hmm.emission_matrix[current_state].get(text[i], 0), prev_state) for prev_state in states)
            V[i][current_state] = prob
            new_path[current_state] = path[state] + [current_state]
        path = new_path

    (prob, state) = max((V[len(text) - 1][state], state) for state in states)
    result = "".join([text[i] + (" " if path[state][i] in "SE" else "") for i in range(len(text))])
    return result.strip()

if __name__ == "__main__":
    generate_state_file()
    sample_text = "今天你学习数学了吗"
    hmm = HMM()
    hmm.train()
    segmented_text = viterbi(sample_text, hmm)
    print(segmented_text)
