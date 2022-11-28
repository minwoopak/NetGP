import torch
from torch import nn
import torch.nn.functional as F


class MultiModalRegressor(nn.Module):
    def __init__(self,
                 drug_embed_net,
                 cell_embed_net,
                 np_score_embed_net,
                 fc_in_dim,
                 fc_hid_dim=[512,256],
                 dropout=0.5
                ):
        super(MultiModalRegressor, self).__init__()
        
        self.drug_embed_net = drug_embed_net
        self.cell_embed_net = cell_embed_net
        self.np_score_embed_net = np_score_embed_net
        
        # === Classifier === #
        self.fc = nn.Linear(fc_in_dim, fc_hid_dim[0])
        self.act = nn.ReLU()
        self.classifier = nn.ModuleList()
        
        for input_size, output_size in zip(fc_hid_dim, fc_hid_dim[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                              nn.BatchNorm1d(output_size),
                              self.act,
                              nn.Dropout(p=dropout)
                             )
            )
        self.fc2 = nn.Linear(fc_hid_dim[-1], 1)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)
    
    def forward(self, drug_x, cell_x, np_score_x):
        # === Embed each input === #
        drug_x = self.drug_embed_net(drug_x)
        cell_x = self.cell_embed_net(cell_x)
        np_score_x = self.np_score_embed_net(np_score_x)
        
        # === Adapt Cell line with NP scores === #
        cell_x = np_score_x * cell_x
        
        # === Concat with Drug === #
        context_vector = torch.cat((drug_x, cell_x), dim=1)
        
        x = F.relu(self.fc(context_vector))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        return output
    
    def get_embedding(self, drug_x, cell_x, np_score_x):
        drug_x = self.drug_embed_net(drug_x)
        cell_x = self.cell_embed_net(cell_x)
        np_score_x = self.np_score_embed_net(np_score_x)
        # === Adapt Cell line with NP scores === #
        weighted_cell_x = np_score_x * cell_x
        
        # === Concat with Drug === #
        context_vector = torch.cat((drug_x, weighted_cell_x), dim=1)
        
        x = F.relu(self.fc(context_vector))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        return drug_x, cell_x, np_score_x, output


class NaiveModel(nn.Module):
    def __init__(self,
                 drug_embed_net,
                 fc_in_dim,
                 fc_hid_dim=[512,512],
                 dropout=0.5
                ):
        super(NaiveModel, self).__init__()
        
        self.drug_embed_net = drug_embed_net
        
        # === Classifier === #
        self.fc = nn.Linear(fc_in_dim, fc_hid_dim[0])
        self.act = nn.ReLU()
        self.classifier = nn.ModuleList()
        
        for input_size, output_size in zip(fc_hid_dim, fc_hid_dim[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                              nn.BatchNorm1d(output_size),
                              self.act,
                              nn.Dropout(p=dropout)
                             )
            )
        self.fc2 = nn.Linear(fc_hid_dim[-1], 1)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)
        
    def forward(self, drug_x, cell_x):
        # === embed drug === #
        drug_x = self.drug_embed_net(drug_x)
        # === concat drug_x, cell_x === #
        input_vector = torch.cat((drug_x, cell_x), dim=1)
        x = F.relu(self.fc(input_vector))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        return output
    
    def get_embedding(self, drug_x, cell_x):
        # === concat drug_x, cell_x === #
        input_vector = torch.cat((drug_x, cell_x), dim=1)
        x = F.relu(self.fc(context_vector))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        return drug_x, cell_x, output
    

class NPDrugResponseNet(nn.Module):
    def __init__(self, #drug_dim, cell_dim, 
                 drug_embed_net, #cell_embed_net, 
                 attention_net, 
                 fc_in_dim=18115+1024, fc_hid_dim=[512, 512], dropout=0.5):
        super(NPDrugResponseNet, self).__init__()
        #self.drug_dim = drug_dim
        #self.cell_dim = cell_dim
        self.fc_hid_dim = fc_hid_dim
        
        self.drug_embed_net = drug_embed_net
#         self.cell_embed_net = cell_embed_net
        self.attention_net = attention_net
        
        # === Classifier === #
        self.fc = nn.Linear(fc_in_dim, self.fc_hid_dim[0])
        self.act = nn.ReLU()
        self.dropout = dropout
        self.classifier = nn.ModuleList()
        
        for input_size, output_size in zip(self.fc_hid_dim, self.fc_hid_dim[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                              nn.BatchNorm1d(output_size),
                              self.act,
                              nn.Dropout(p=self.dropout)
                             )
            )
        self.fc2 = nn.Linear(self.fc_hid_dim[-1], 1)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)
        
    def forward(self, drug_x, cell_x, np_scores):
        # === embed drug === #
        drug_x = self.drug_embed_net(drug_x)
#         # === embed cell line === #
#         cell_x = self.cell_embed_net(cell_x)
        # === obtain attention from drug embedding === #
        c_weights = self.attention_net(np_scores)
        # === apply attention to cell line embedding === #
        cell_x = c_weights * cell_x
        context_vector = torch.cat((drug_x, cell_x), dim=1)
        x = F.relu(self.fc(context_vector))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        return output#, drug_x, c_weights
    
    def get_embedding(self, drug_x, cell_x, np_scores):
        # === embed drug === #
        drug_x = self.drug_embed_net(drug_x)
        # === obtain attention from drug embedding === #
        c_weights = self.attention_net(np_scores)
        # === apply attention to cell line embedding === #
        cell_x = c_weights * cell_x
        context_vector = torch.cat((drug_x, cell_x), dim=1)
        # === return drug_embedding, attention_weights, concat_context_vector === #
        return drug_x, c_weights, context_vector
    

class NPscorePass(nn.Module):
    def __init__(self):
        super(NPscorePass, self).__init__()
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, np_scores):
        #x = self.sigmoid(x)
        #np_scores += 1
        return np_scores
    
    
class DrugResponseNet(nn.Module):
    def __init__(self, #drug_dim, cell_dim, 
                 drug_embed_net, #cell_embed_net, 
                 attention_net, 
                 fc_in_dim=18115+1024, fc_hid_dim=[512, 512], dropout=0.5):
        super(DrugResponseNet, self).__init__()
        #self.drug_dim = drug_dim
        #self.cell_dim = cell_dim
        self.fc_hid_dim = fc_hid_dim
        
        self.drug_embed_net = drug_embed_net
#         self.cell_embed_net = cell_embed_net
        self.attention_net = attention_net
        
        # === Classifier === #
        self.fc = nn.Linear(fc_in_dim, self.fc_hid_dim[0])
        self.act = nn.ReLU()
        self.dropout = dropout
        self.classifier = nn.ModuleList()
        
        for input_size, output_size in zip(self.fc_hid_dim, self.fc_hid_dim[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                              nn.BatchNorm1d(output_size),
                              self.act,
                              nn.Dropout(p=self.dropout)
                             )
            )
        self.fc2 = nn.Linear(self.fc_hid_dim[-1], 1)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)
        
    def forward(self, drug_x, cell_x):
        # === embed drug === #
        drug_x = self.drug_embed_net(drug_x)
#         # === embed cell line === #
#         cell_x = self.cell_embed_net(cell_x)
        # === obtain attention from drug embedding === #
        c_weights = self.attention_net(drug_x)
        # === apply attention to cell line embedding === #
        cell_x = c_weights * cell_x
        context_vector = torch.cat((drug_x, cell_x), dim=1)
        x = F.relu(self.fc(context_vector))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        return output#, drug_x, c_weights
    
    def get_embedding(self, drug_x, cell_x):
        # === embed drug === #
        drug_x = self.drug_embed_net(drug_x)
        # === obtain attention from drug embedding === #
        c_weights = self.attention_net(drug_x)
        # === apply attention to cell line embedding === #
        cell_x = c_weights * cell_x
        context_vector = torch.cat((drug_x, cell_x), dim=1)
        x = F.relu(self.fc(context_vector))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        # === return drug_embedding, attention_weights, concat_context_vector === #
        return drug_x, c_weights, context_vector, output
    

class ResponseUnit(nn.Module):
    def __init__(self, drug_dim=1024, fc_hid_dim=[512, 512], output_dim=18115, dropout=0.5):
        super(ResponseUnit, self).__init__()
        self.fc_hid_dim = fc_hid_dim
        self.fc = nn.Linear(drug_dim, self.fc_hid_dim[0])
        self.act = nn.ReLU()
        self.dropout = dropout
        self.classifier = nn.ModuleList()
        
        for input_size, output_size in zip(self.fc_hid_dim, self.fc_hid_dim[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                              nn.BatchNorm1d(output_size),
                              self.act,
                              nn.Dropout(p=self.dropout)
                             )
            )
        self.fc2 = nn.Linear(self.fc_hid_dim[-1], output_dim)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)
    
    def forward(self, x):
        x = F.relu(self.fc(x))
        for fc in self.classifier:
            x = fc(x)
        x = torch.sigmoid(self.fc2(x))
        return x
    
    
class NPResponseUnitNet(nn.Module):
    def __init__(self, #drug_dim, cell_dim, 
                 drug_embed_net, #cell_embed_net, 
                 attention_net, 
                 fc_in_dim=18115+1024, fc_hid_dim=[512, 512], dropout=0.5):
        super(NPResponseUnitNet, self).__init__()
        self.fc_hid_dim = fc_hid_dim
        
        self.drug_embed_net = drug_embed_net
        self.attention_net = attention_net
        
        # === Classifier === #
        self.fc = nn.Linear(fc_in_dim, self.fc_hid_dim[0])
        self.act = nn.ReLU()
        self.dropout = dropout
        self.classifier = nn.ModuleList()
        
        for input_size, output_size in zip(self.fc_hid_dim, self.fc_hid_dim[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                              nn.BatchNorm1d(output_size),
                              self.act,
                              nn.Dropout(p=self.dropout)
                             )
            )
        self.fc2 = nn.Linear(self.fc_hid_dim[-1], 1)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)
        
    def forward(self, drug_x, cell_x, np_scores):
        # === embed drug === #
        drug_x = self.drug_embed_net(drug_x)
        # === weight cell line exprs by np scores === #
        cell_x = cell_x * np_scores
        # === obtain attention from drug embedding === #
        c_weights = self.attention_net(drug_x)
        # === apply attention to cell line embedding === #
        cell_x = cell_x * c_weights
        context_vector = torch.cat((drug_x, cell_x), dim=1)
        x = F.relu(self.fc(context_vector))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        return output#, drug_x, c_weights
    
    def get_embedding(self, drug_x, cell_x, np_scores):
        # === embed drug === #
        drug_x = self.drug_embed_net(drug_x)
        # === weight cell line exprs by np scores === #
        cell_x = cell_x * np_scores
        # === obtain attention from drug embedding === #
        c_weights = self.attention_net(drug_x)
        # === apply attention to cell line embedding === #
        cell_x = c_weights * cell_x
        context_vector = torch.cat((drug_x, cell_x), dim=1)
        # === return drug_embedding, attention_weights, concat_context_vector === #
        return drug_x, cell_x, c_weights, context_vector

    
class NPBiasNet(nn.Module):
    def __init__(self, #drug_dim, cell_dim, 
                 drug_embed_net, #cell_embed_net, 
                 attention_net, 
                 fc_in_dim=18115+1024, fc_hid_dim=[512, 512], dropout=0.5):
        super(NPBiasNet, self).__init__()
        self.fc_hid_dim = fc_hid_dim
        
        self.drug_embed_net = drug_embed_net
        self.attention_net = attention_net
        
        # === Classifier === #
        self.fc = nn.Linear(fc_in_dim, self.fc_hid_dim[0])
        self.act = nn.ReLU()
        self.dropout = dropout
        self.classifier = nn.ModuleList()
        
        for input_size, output_size in zip(self.fc_hid_dim, self.fc_hid_dim[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                              nn.BatchNorm1d(output_size),
                              self.act,
                              nn.Dropout(p=self.dropout)
                             )
            )
        self.fc2 = nn.Linear(self.fc_hid_dim[-1], 1)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)
        
    def forward(self, drug_x, cell_x, np_scores):
        # === embed drug === #
        drug_x = self.drug_embed_net(drug_x)
        # === obtain attention from drug embedding === #
        c_weights = self.attention_net(np_scores)
        # === apply attention to cell line embedding === #
        cell_x = c_weights + cell_x
        context_vector = torch.cat((drug_x, cell_x), dim=1)
        x = F.relu(self.fc(context_vector))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        return output
    
    def get_embedding(self, drug_x, cell_x, np_scores):
        # === embed drug === #
        drug_x = self.drug_embed_net(drug_x)
        # === obtain attention from drug embedding === #
        c_weights = self.attention_net(np_scores)
        # === apply attention to cell line embedding === #
        cell_x = c_weights + cell_x
        context_vector = torch.cat((drug_x, cell_x), dim=1)
        # === return drug_embedding, attention_weights, concat_context_vector === #
        return drug_x, c_weights, context_vector