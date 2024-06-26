import pandas as pd
import numpy as np
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import f1_score, mean_squared_error
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# Fixing the randomness of CUDA.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fixing random state.
np.random.seed(42)
torch.manual_seed(42)

# Checking if CUDA is available.
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch Version : {}".format(torch.__version__))
print(DEVICE)

# Set workspace and model name.
# worksapce: main fold of LIAR2023
worksapce = ''
# worksapce = 'E:\LIAR2023/'
model_name = '1.9.justification'
model_save = model_name + '.pt'

print(model_save)

# Set hyperparameters for training.
num_epochs = 20
batch_size = 32
learning_rate = 1e-3
num_classes = 6
padding_idx = 0
dropout = 0.5

# Dimension of news and justification
embedding_dim = 128

# Dimension of each textual context feature
metadata_each_dim = 10

# Dimension of input numerical context sequences.
input_dim_metadata = 6

# Output dimension of each pipeline
output_dim = 6

# Dimension of input textual context sequences.
# number_of_textual_context_feature * metadata_each_dim
input_dim = 6 * metadata_each_dim

# Vocabulary size of tokenizer for transform text data into sequences.
vocab_size = 30522

# Set hyperparameters for TextCNN.
n_filters = 128
filter_sizes = [3,4,5]

# Set hyperparameters for Bi-LSTM.
hidden_dim = 64
n_layers = 1
bidirectional = True


# col = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']
col = ["id", "label", "statement", "date", "subject", "speaker", "speaker_description", "state_info", "true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts", "context", "justification"]

label_map = {0: 'pants-fire', 1: 'false', 2: 'barely-true', 3: 'half-true', 4: 'mostly-true', 5: 'true'}
label_convert = {'pants-fire': 0, 'false': 1, 'barely-true': 2, 'half-true': 3, 'mostly-true': 4, 'true':5}

train_data = pd.read_csv(worksapce + 'train.csv')
test_data = pd.read_csv(worksapce + 'test.csv')
val_data = pd.read_csv(worksapce + 'valid.csv')

# Replace NaN values with 'NaN'
train_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]] = train_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]].fillna(0)
train_data.fillna('NaN', inplace=True)

test_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]] = test_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]].fillna(0)
test_data.fillna('NaN', inplace=True)

val_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]] = val_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]].fillna(0)
val_data.fillna('NaN', inplace=True)

# Tokenizing text
def textProcess(input_text, max_length = -1):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if max_length == -1:
        tokens = tokenizer(input_text, truncation=True, padding=True)
    else:
        tokens = tokenizer(input_text, truncation=True, padding='max_length', max_length=max_length)
    return tokens


# Define a custom dataset for loading the data
class LiarDataset(data.Dataset):
    def __init__(self, data_df, statement, label_onehot, label, date, subject, speaker, speaker_description, state_info,
                    true_counts, mostly_true_counts, half_true_counts, mostly_false_counts,
                    false_counts, pants_on_fire_counts, context, justification):
        self.data_df = data_df
        self.statement = statement
        self.label_onehot = label_onehot
        self.label = label
        self.justification = justification
        self.metadata_text = torch.cat((date.int(), subject.int(), speaker.int(), speaker_description.int(), state_info.int(), context.int()), dim=-1)
        self.metadata_number = torch.cat((torch.tensor(true_counts, dtype=torch.float).unsqueeze(1), torch.tensor(mostly_true_counts, dtype=torch.float).unsqueeze(1), 
                                   torch.tensor(half_true_counts, dtype=torch.float).unsqueeze(1), torch.tensor(mostly_false_counts, dtype=torch.float).unsqueeze(1), 
                                   torch.tensor(false_counts, dtype=torch.float).unsqueeze(1), torch.tensor(pants_on_fire_counts, dtype=torch.float).unsqueeze(1)), dim=-1)

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        statement = self.statement[idx]
        label_onehot = self.label_onehot[idx]
        label = self.label[idx]
        justification = self.justification[idx]
        metadata_text = self.metadata_text[idx]
        metadata_number = self.metadata_number[idx]
        return statement, label_onehot, label, metadata_text, metadata_number, justification


# Define the data loaders for training and validation
train_text = torch.tensor(textProcess(train_data['statement'].tolist())['input_ids'])
train_justification = torch.tensor(textProcess(train_data['justification'].tolist())['input_ids'])
train_label = torch.nn.functional.one_hot(torch.tensor(train_data['label'].replace(label_convert)), num_classes=6).type(torch.float64)
train_date = torch.tensor(textProcess(train_data['date'].tolist(), metadata_each_dim)['input_ids'])
train_subject = torch.tensor(textProcess(train_data['subject'].tolist(), metadata_each_dim)['input_ids'])
train_speaker = torch.tensor(textProcess(train_data['speaker'].tolist(), metadata_each_dim)['input_ids'])
train_speaker_description = torch.tensor(textProcess(train_data['speaker_description'].tolist(), metadata_each_dim)['input_ids'])
train_state_info = torch.tensor(textProcess(train_data['state_info'].tolist(), metadata_each_dim)['input_ids'])
train_context = torch.tensor(textProcess(train_data['context'].tolist(), metadata_each_dim)['input_ids'])

train_dataset = LiarDataset(train_data, train_text, train_label, torch.tensor(train_data['label'].replace(label_convert)), 
                            train_date, train_subject, train_speaker, train_speaker_description, train_state_info, 
                            train_data['true_counts'].tolist(), train_data['mostly_true_counts'].tolist(), 
                            train_data['half_true_counts'].tolist(), train_data['mostly_false_counts'].tolist(), 
                            train_data['false_counts'].tolist(), train_data['pants_on_fire_counts'].tolist(), train_context, train_justification)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_text = torch.tensor(textProcess(val_data['statement'].tolist())['input_ids'])
val_justification = torch.tensor(textProcess(val_data['justification'].tolist())['input_ids'])
val_label = torch.nn.functional.one_hot(torch.tensor(val_data['label'].replace(label_convert)), num_classes=6).type(torch.float64)
val_date = torch.tensor(textProcess(val_data['date'].tolist(), metadata_each_dim)['input_ids'])
val_subject = torch.tensor(textProcess(val_data['subject'].tolist(), metadata_each_dim)['input_ids'])
val_speaker = torch.tensor(textProcess(val_data['speaker'].tolist(), metadata_each_dim)['input_ids'])
val_speaker_description = torch.tensor(textProcess(val_data['speaker_description'].tolist(), metadata_each_dim)['input_ids'])
val_state_info = torch.tensor(textProcess(val_data['state_info'].tolist(), metadata_each_dim)['input_ids'])
val_context = torch.tensor(textProcess(val_data['context'].tolist(), metadata_each_dim)['input_ids'])

val_dataset = LiarDataset(val_data, val_text, val_label, torch.tensor(val_data['label'].replace(label_convert)),
                          val_date, val_subject, val_speaker, val_speaker_description, val_state_info, 
                          val_data['true_counts'].tolist(), val_data['mostly_true_counts'].tolist(), 
                          val_data['half_true_counts'].tolist(), val_data['mostly_false_counts'].tolist(), 
                          val_data['false_counts'].tolist(), val_data['pants_on_fire_counts'].tolist(), val_context, val_justification)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size)

test_text = torch.tensor(textProcess(test_data['statement'].tolist())['input_ids'])
test_justification = torch.tensor(textProcess(test_data['justification'].tolist())['input_ids'])
test_label = torch.nn.functional.one_hot(torch.tensor(test_data['label'].replace(label_convert)), num_classes=6).type(torch.float64)
test_date = torch.tensor(textProcess(test_data['date'].tolist(), metadata_each_dim)['input_ids'])
test_subject = torch.tensor(textProcess(test_data['subject'].tolist(), metadata_each_dim)['input_ids'])
test_speaker = torch.tensor(textProcess(test_data['speaker'].tolist(), metadata_each_dim)['input_ids'])
test_speaker_description = torch.tensor(textProcess(test_data['speaker_description'].tolist(), metadata_each_dim)['input_ids'])
test_state_info = torch.tensor(textProcess(test_data['state_info'].tolist(), metadata_each_dim)['input_ids'])
test_context = torch.tensor(textProcess(test_data['context'].tolist(), metadata_each_dim)['input_ids'])

test_dataset = LiarDataset(test_data, test_text, test_label, torch.tensor(test_data['label'].replace(label_convert)),
                          test_date, test_subject, test_speaker, test_speaker_description, test_state_info, 
                          test_data['true_counts'].tolist(), test_data['mostly_true_counts'].tolist(), 
                          test_data['half_true_counts'].tolist(), test_data['mostly_false_counts'].tolist(), 
                          test_data['false_counts'].tolist(), test_data['pants_on_fire_counts'].tolist(), test_context, test_justification)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size)


class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, membership_num):
        super(FuzzyLayer, self).__init__()

        # input_dim: feature number of the dataset
        # membership_num: number of membership function, also known as the class number

        self.input_dim = input_dim
        self.membership_num = membership_num

        self.membership_miu = nn.Parameter(torch.Tensor(self.membership_num, self.input_dim).to(DEVICE), requires_grad=True)
        self.membership_sigma = nn.Parameter(torch.Tensor(self.membership_num, self.input_dim).to(DEVICE), requires_grad=True)

        nn.init.xavier_uniform_(self.membership_miu)
        nn.init.ones_(self.membership_sigma)

    def forward(self, input_seq):
        batch_size = input_seq.size()[0]
        input_seq_exp = input_seq.unsqueeze(1).expand(batch_size, self.membership_num, self.input_dim)
        membership_miu_exp = self.membership_miu.unsqueeze(0).expand(batch_size, self.membership_num, self.input_dim)
        membership_sigma_exp = self.membership_sigma.unsqueeze(0).expand(batch_size, self.membership_num, self.input_dim)

        fuzzy_membership = torch.mean(torch.exp((-1 / 2) * ((input_seq_exp - membership_miu_exp) / membership_sigma_exp) ** 2), dim=-1)
        return fuzzy_membership



class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embedding_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)
        #embedded = [batch size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=1)
        self.rnn = nn.LSTM(32, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, metadata):
        #metadata = [batch size, metadata dim]

        embedded = self.dropout(self.embedding(metadata))
        #embedded = [batch size, metadata dim, emb dim]

        embedded = torch.reshape(embedded, (metadata.size(0), 128, 1))

        conved = F.relu(self.conv(embedded))
        #conved = [batch size, n_filters, metadata dim - filter_sizes[n] + 1]

        conved = torch.reshape(conved, (metadata.size(0), 32))

        outputs, (hidden, cell) = self.rnn(conved)
        #outputs = [metadata dim - filter_sizes[n] + 1, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        # hidden = self.dropout(torch.cat((hidden[-1,:], hidden[0,:]), dim = -1))
        #hidden = [batch size, hid dim * num directions]

        return self.fc(outputs)


class LiarModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx, input_dim, input_dim_metadata, hidden_dim, n_layers, bidirectional):
        super().__init__()

        self.textcnn = TextCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx)
        self.justification_cnn = TextCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx)
        self.textcnn2 = TextCNN(vocab_size, input_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx)
        self.cnn_bilstm = CNNBiLSTM(input_dim_metadata, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
        self.fuzzy = FuzzyLayer(output_dim, output_dim)
        self.fuse = nn.Linear(output_dim * 1, output_dim)
        # self.fuse = FuzzyLayer(output_dim * 5, output_dim)
    
    def forward(self, text, metadata_text, metadata_number, justification):
        #text = [batch size, sent len]
        #metadata = [batch size, metadata dim]

        # text_output = self.textcnn(text)
        # metadata_output_text = self.textcnn2(metadata_text)
        # metadata_output_number = self.cnn_bilstm(metadata_number)
        # metadata_output_number_fuzzy = self.fuzzy(metadata_output_number)
        justification_output = self.justification_cnn(justification)

        fused_output = self.fuse(justification_output)

        return fused_output


model = LiarModel(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx, input_dim, input_dim_metadata, hidden_dim, n_layers, bidirectional).to(DEVICE)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

# Record the training process
Train_acc = []
Train_loss = []
Train_macro_f1 = []
Train_micro_f1 = []
Train_mse = []
Train_rmse = []

Val_acc = []
Val_loss = []
Val_macro_f1 = []
Val_micro_f1 = []
Val_mse = []
Val_rmse = []

Test_acc = []
Test_loss = []
Test_macro_f1 = []
Test_micro_f1 = []
Test_mse = []
Test_rmse = []

def train(num_epochs, model, train_loader, val_loader, optimizer, criterion, model_save):
    epoch_trained = 0
    train_label_all = []
    train_label_onehot_all = []
    train_predict_all = []
    train_predict_onehot_all = []

    val_label_all = []
    val_label_onehot_all = []
    val_predict_all = []
    val_predict_onehot_all = []
    best_valid_loss = float('inf')

    test_label_all = []
    test_label_onehot_all = []
    test_predict_all = []
    test_predict_onehot_all = []

    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_trained += 1
        epoch_start_time = time.time()
        # Training
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for statements, label_onehot, label, metadata_text, metadata_number, justification in train_loader:
            statements = statements.to(DEVICE)
            label_onehot = label_onehot.to(DEVICE)
            label = label.to(DEVICE)
            metadata_text = metadata_text.to(DEVICE)
            metadata_number = metadata_number.to(DEVICE)
            justification = justification.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(statements, metadata_text, metadata_number, justification)
            loss = criterion(outputs, label_onehot)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, train_predicted = torch.max(outputs, 1)
            train_accuracy += sum(train_predicted == label)
            train_predict_all += train_predicted.tolist()
            train_predict_onehot_all += outputs.tolist()
            train_label_all += label.tolist()
            train_label_onehot_all += label_onehot.tolist()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader.dataset)
        train_macro_f1 = f1_score(train_label_all, train_predict_all, average='macro')
        train_micro_f1 = f1_score(train_label_all, train_predict_all, average='micro')
        train_mse = mean_squared_error(train_label_all, train_predict_all)
        train_rmse = np.sqrt(train_mse)

        Train_acc.append(train_accuracy.tolist())
        Train_loss.append(train_loss)
        Train_macro_f1.append(train_macro_f1)
        Train_micro_f1.append(train_micro_f1)
        Train_mse.append(train_mse)
        Train_rmse.append(train_rmse)

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for statements, label_onehot, label, metadata_text, metadata_number, justification in val_loader:
                statements = statements.to(DEVICE)
                label_onehot = label_onehot.to(DEVICE)
                label = label.to(DEVICE)
                metadata_text = metadata_text.to(DEVICE)
                metadata_number = metadata_number.to(DEVICE)
                justification = justification.to(DEVICE)

                val_outputs = model(statements, metadata_text, metadata_number, justification)
                val_loss += criterion(val_outputs, label_onehot).item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_accuracy += sum(val_predicted == label)
                val_predict_all += val_predicted.tolist()
                val_predict_onehot_all += val_outputs.tolist()
                val_label_all += label.tolist()
                val_label_onehot_all += label_onehot.tolist()

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader.dataset)
        val_macro_f1 = f1_score(val_label_all, val_predict_all, average='macro')
        val_micro_f1 = f1_score(val_label_all, val_predict_all, average='micro')
        val_mse = mean_squared_error(val_label_all, val_predict_all)
        val_rmse = np.sqrt(val_mse)

        Val_acc.append(val_accuracy.tolist())
        Val_loss.append(val_loss)
        Val_macro_f1.append(val_macro_f1)
        Val_micro_f1.append(val_micro_f1)
        Val_mse.append(val_mse)
        Val_rmse.append(val_rmse)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), model_save)
            print(f'***** Best Result Updated at Epoch {epoch_trained}, Val Loss: {val_loss:.4f} *****')

        # Testing
        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for statements, label_onehot, label, metadata_text, metadata_number, justification in test_loader:
                statements = statements.to(DEVICE)
                label_onehot = label_onehot.to(DEVICE)
                label = label.to(DEVICE)
                metadata_text = metadata_text.to(DEVICE)
                metadata_number = metadata_number.to(DEVICE)
                justification = justification.to(DEVICE)

                test_outputs = model(statements, metadata_text, metadata_number, justification)
                test_loss += criterion(test_outputs, label_onehot).item()
                _, test_predicted = torch.max(test_outputs, 1)
                test_accuracy += sum(test_predicted == label)
                test_predict_all += test_predicted.tolist()
                test_predict_onehot_all += test_outputs.tolist()
                test_label_all += label.tolist()
                test_label_onehot_all += label_onehot.tolist()

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader.dataset)
        test_macro_f1 = f1_score(test_label_all, test_predict_all, average='macro')
        test_micro_f1 = f1_score(test_label_all, test_predict_all, average='micro')
        test_mse = mean_squared_error(test_label_all, test_predict_all)
        test_rmse = np.sqrt(test_mse)

        Test_acc.append(test_accuracy.tolist())
        Test_loss.append(test_loss)
        Test_macro_f1.append(test_macro_f1)
        Test_micro_f1.append(test_micro_f1)
        Test_mse.append(test_mse)
        Test_rmse.append(test_rmse)

        # Print the losses and accuracy
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        print(f"Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, \nTrain Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1 Macro: {train_macro_f1:.4f}, Train F1 Micro: {train_micro_f1:.4f}, Train RMSE: {train_rmse:.4f}, \nVal Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1 Macro: {val_macro_f1:.4f}, Val F1 Micro: {val_micro_f1:.4f}, Val RMSE: {val_rmse:.4f}, \nTest Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test F1 Macro: {test_macro_f1:.4f}, Test F1 Micro: {test_micro_f1:.4f}, Test RMSE: {test_rmse:.4f}\n")

    end_time = time.time()
    training_time = end_time - start_time
    print(f'Total Training Time: {training_time:.2f}s')


train(num_epochs, model, train_loader, val_loader, optimizer, criterion, model_save)


# Evaluate the model on new data
def test(model, test_loader, model_save):
    model.load_state_dict(torch.load(model_save))
    model.eval()

    test_label_all = []
    test_label_onehot_all = []
    test_predict_all = []
    test_predict_onehot_all = []
    
    test_loss = 0.0
    test_accuracy = 0.0
    with torch.no_grad():
        for statements, label_onehot, label, metadata_text, metadata_number, justification in test_loader:
            statements = statements.to(DEVICE)
            label_onehot = label_onehot.to(DEVICE)
            label = label.to(DEVICE)
            metadata_text = metadata_text.to(DEVICE)
            metadata_number = metadata_number.to(DEVICE)
            justification = justification.to(DEVICE)

            test_outputs = model(statements, metadata_text, metadata_number, justification)
            test_loss += criterion(test_outputs, label_onehot).item()
            _, test_predicted = torch.max(test_outputs, 1)
            
            test_accuracy += sum(test_predicted == label)
            test_predict_all += test_predicted.tolist()
            test_predict_onehot_all += test_outputs.tolist()
            test_label_all += label.tolist()
            test_label_onehot_all += label_onehot.tolist()

    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader.dataset)
    test_macro_f1 = f1_score(test_label_all, test_predict_all, average='macro')
    test_micro_f1 = f1_score(test_label_all, test_predict_all, average='micro')
    test_mse = mean_squared_error(test_label_all, test_predict_all)
    test_rmse = np.sqrt(test_mse)

    print(f'\nTest Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test F1 Macro: {test_macro_f1:.4f}, Test F1 Micro: {test_micro_f1:.4f}, Test RMSE: {test_rmse:.4f}')

test(model, test_loader, model_save)