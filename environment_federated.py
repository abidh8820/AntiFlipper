from __future__ import print_function
from lib2to3.pgen2.tokenize import tokenize
import numpy as np
import torch
import inspect

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import *
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from models import *
from utils import *
from sampling import *
from datasets import *
import os
import random
from tqdm import tqdm_notebook
import copy
from operator import itemgetter
import time
from random import shuffle
from aggregation import *
from IPython.display import clear_output
import gc




class Peer():
    # Class variable shared among all the instances
    _performed_attacks = 0
    @property
    def performed_attacks(self):
        return type(self)._performed_attacks

    @performed_attacks.setter
    def performed_attacks(self,val):
        type(self)._performed_attacks = val

    def __init__(self, peer_id, peer_pseudonym, local_data, labels, criterion, 
                device, local_epochs, local_bs, local_lr, 
                local_momentum, peer_type = 'honest'):

        self.peer_id = peer_id
        self.peer_pseudonym = peer_pseudonym
        self.local_data = local_data
        self.labels = labels
        self.criterion = criterion
        self.device = device
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.peer_type = peer_type
#======================================= Start of training function ===========================================================#
        



    



    def participant_update(self, global_epoch, model, attack_type = 'no_attack', malicious_behavior_rate = 0, 
                            source_class = None, target_class = None, dataset_name = None) :
        
        epochs = self.local_epochs
        train_loader = DataLoader(self.local_data, self.local_bs, shuffle = True, drop_last=True)
        attacked = 0
        #print(f"Inside Peer Update : attack type = {attack_type} and peer type = {self.peer_type}")

        #Get the poisoned training data of the peer in case of label-flipping or backdoor attacks
        if (attack_type == 'label_flipping') and (self.peer_type == 'attacker'):
            r = np.random.random()
            if r <= malicious_behavior_rate or True:
                if dataset_name != 'IMDB':
                    poisoned_data = label_filp(self.local_data, source_class, target_class)
                    train_loader = DataLoader(poisoned_data, self.local_bs, shuffle = True, drop_last=True)
                self.performed_attacks+=1
                
                attacked = 1
                #print('Label flipping attack launched by', self.peer_pseudonym, 'to flip class ', source_class,
                #' to class ', target_class)
                #print('Label flipping attack launched by', self.peer_id, 'to flip classes are randomly flipped ')
        


        lr=self.local_lr
    
        if dataset_name == 'IMDB':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr,
                        momentum=self.local_momentum, weight_decay=5e-4)
        model.train()
        epoch_loss = []
        peer_grad = []
        t = 0

        model_global = copy.deepcopy(model)

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if dataset_name == 'IMDB':
                    target = target.view(-1,1) * (1 - attacked)

                data, target = data.to(self.device), target.to(self.device)
                # for CIFAR10 multi-LF attack
                # if attacked:
                #     target = (target + 1)%10
                output = model(data)


                # criterion is a loss function named CrossEntropyLoss
                # loss is a tensor here
                loss = self.criterion(output, target)
                #backward propagation using the tensor
                loss.backward()    


                #converting the tensor into a scaler
                epoch_loss.append(loss.item())
                
                # get gradients
                cur_time = time.time()
                for i, (name, params) in enumerate(model.named_parameters()):
                    if params.requires_grad:
                        if epoch == 0 and batch_idx == 0:
                            peer_grad.append(params.grad.clone())
                        else:
                            peer_grad[i]+= params.grad.clone()   
                t+= time.time() - cur_time    
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
               
        # print('Train epoch: {} \tLoss: {:.6f}'.format((epochs+1), np.mean(epoch_loss)))
    
        if (attack_type == 'gaussian' and self.peer_type == 'attacker'):
            update, flag =  gaussian_attack(model.state_dict(), self.peer_pseudonym,
            malicious_behavior_rate = malicious_behavior_rate, device = self.device)
            if flag == 1:
                self.performed_attacks+=1
                attacked = 1
            model.load_state_dict(update)

        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print("Number of Attacks:{}".format(self.performed_attacks))
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        model = model.cpu()

        ### entirely local model loss


        #initializing the local model after updaates

        #print(type(peer_grad), type(peer_grad[0]), type(peer_grad[0].item()), type(model.state_dict()))
        globalAccuracy = []
        model_local  = copy.deepcopy(model)

        # runs of gpu
        model_global = model_global.cuda()
        model_local = model_local.cuda()
        
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if dataset_name == 'IMDB':
                    target = target.view(-1,1) * (1 - attacked)

                data, target = data.to(self.device), target.to(self.device)

                globalModelAcc = AashnanAccuracy(model_global, data,target.tolist())
                globalAccuracy.append(globalModelAcc)

        model_global = model_global.cpu()
        model_local = model_local.cpu()
        # at the last of the return values loss on local model and previous global model is returned
        return model.state_dict(), peer_grad , model, np.mean(epoch_loss), attacked, t, np.mean(globalAccuracy)
#======================================= End of training function =============================================================#
#========================================= End of Peer class ====================================================================


class FL:
    def __init__(self, dataset_name, model_name, dd_type, num_peers, frac_peers, 
    seed, test_batch_size, criterion, global_rounds, local_epochs, local_bs, local_lr,
    local_momentum, labels_dict, device, attackers_ratio = 0,
    class_per_peer=2, samples_per_class= 250, rate_unbalance = 1, alpha = 1,source_class = None):

        FL._history = np.zeros(num_peers)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_peers = num_peers
        self.peers_pseudonyms = ['Peer ' + str(i+1) for i in range(self.num_peers)]
        self.frac_peers = frac_peers
        self.seed = seed
        self.test_batch_size = test_batch_size
        self.criterion = criterion
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.labels_dict = labels_dict
        self.num_classes = len(self.labels_dict)
        self.device = device
        self.attackers_ratio = attackers_ratio
        self.class_per_peer = class_per_peer
        self.samples_per_class = samples_per_class
        self.rate_unbalance = rate_unbalance
        self.source_class = source_class
        self.dd_type = dd_type
        self.alpha = alpha
        self.embedding_dim = 100
        self.peers = []
        self.trainset, self.testset = None, None
        
        # Fix the random state of the environment
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
       
        #Loading of data
        module = inspect.getmodule(distribute_dataset)
        if module:
            print(f"The function {distribute_dataset.__name__} is from the module {module.__name__}.")
            
        self.trainset, self.testset, user_groups_train, tokenizer = distribute_dataset(self.dataset_name, self.num_peers, self.num_classes, 
        self.dd_type, self.class_per_peer, self.samples_per_class, self.alpha)

        self.test_loader = DataLoader(self.testset, batch_size = self.test_batch_size,
            shuffle = False, num_workers = 1)
    
        #Creating model
        self.global_model = setup_model(model_architecture = self.model_name, num_classes = self.num_classes, 
        tokenizer = tokenizer, embedding_dim = self.embedding_dim)
        self.global_model = self.global_model.to(self.device)
        
        # Dividing the training set among peers
        self.local_data = []
        self.have_source_class = []
        self.labels = []
        print('--> Distributing training data among peers')
        for p in user_groups_train:
            self.labels.append(user_groups_train[p]['labels'])
            indices = user_groups_train[p]['data']
            peer_data = CustomDataset(self.trainset, indices=indices)
            self.local_data.append(peer_data)
            if  self.source_class in user_groups_train[p]['labels']:
                 self.have_source_class.append(p)
        print('--> Training data have been distributed among peers')

        # Creating peers instances
        print('--> Creating peers instances')
        m_ = 1

        if self.attackers_ratio > 0:
            #pick m random participants from the workers list
            # k_src = len(self.have_source_class)
            # print('# of peers who have source class examples:', k_src)
            m_ = int(self.attackers_ratio * self.num_peers)
            self.num_attackers = copy.deepcopy(m_)

        peers = list(np.arange(self.num_peers))  
        random.shuffle(peers)
        for i in peers:
            if m_ > 0 :
                self.peers.append(Peer(i, self.peers_pseudonyms[i], 
                self.local_data[i], self.labels[i],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum, peer_type = 'attacker'))
                m_-= 1
            else:
                self.peers.append(Peer(i, self.peers_pseudonyms[i], 
                self.local_data[i], self.labels[i],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum))  

        del self.local_data

#======================================= Start of testning function ===========================================================#
    def test(self, model, device, test_loader, dataset_name = None):
        model.eval()
        test_loss = []
        correct = 0
        n = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            if dataset_name == 'IMDB':
                test_loss.append(self.criterion(output, target.view(-1,1)).item()) # sum up batch loss
                pred = output > 0.5 # get the index of the max log-probability
                correct+= pred.eq(target.view_as(pred)).sum().item()
            else:
                test_loss.append(self.criterion(output, target).item()) # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct+= pred.eq(target.view_as(pred)).sum().item()

            n+= target.shape[0]
        test_loss = np.mean(test_loss)
        print('\nAverage test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, n,
           100*correct / n))
        return  100.0*(float(correct) / n), test_loss
    #======================================= End of testning function =============================================================#
#Test label prediction function    
    def test_label_predictions(self, model, device, test_loader, dataset_name = None):
        model.eval()
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                if dataset_name == 'IMDB':
                    prediction = output > 0.5
                else:
                    prediction = output.argmax(dim=1, keepdim=True)
                
                actuals.extend(target.view_as(prediction))
                predictions.extend(prediction)
        return [i.item() for i in actuals], [i.item() for i in predictions]
    
    #choose random set of peers
    def choose_peers(self):
        #pick m random peers from the available list of peers
        m = max(int(self.frac_peers * self.num_peers), 1)
        selected_peers = np.random.choice(range(self.num_peers), m, replace=False)

        # print('\nSelected Peers\n')
        # for i, p in enumerate(selected_peers):
        #     print(i+1, ': ', self.peers[p].peer_pseudonym, ' is ', self.peers[p].peer_type)
        return selected_peers

        
    def run_experiment(self, attack_type = 'no_attack', malicious_behavior_rate = 0,
        source_class = None, target_class = None, rule = 'fedavg', resume = False, attackers_ratio = 0, dd_type = None):
        simulation_model = copy.deepcopy(self.global_model)
        print('\n===>Simulation started...')
        lfd = LFD(self.num_classes)
        fg = FoolsGold(self.num_peers)
        tolpegin = Tolpegin()
        # copy weights
        global_weights = simulation_model.state_dict()
        last10_updates = []
        test_losses = []
        global_accuracies = []
        source_class_accuracies = []
        cpu_runtimes = []
        noise_scalar = 1.0
        # best_accuracy = 0.0
        mapping = {'honest': 'Good update', 'attacker': 'Bad update'}
        #initializing the trustworthiness of the peers as 1
        trust_w = {}
        bad_nodes  = set()
        bad_cnt = {}  
        
        for xx in range(self.num_peers):
            trust_w[xx]  = 1.0
            bad_cnt[xx] = 0

        normalize(trust_w, bad_nodes)
        #start training
        start_round = 0
        malicious_nodes  = []
        if resume:
            print('Loading last saved checkpoint..')
            checkpoint = torch.load('./checkpoints/'+ self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7')
            simulation_model.load_state_dict(checkpoint['state_dict'])
            start_round = checkpoint['epoch'] + 1
            last10_updates = checkpoint['last10_updates']
            test_losses = checkpoint['test_losses']
            global_accuracies = checkpoint['global_accuracies']
            source_class_accuracies = checkpoint['source_class_accuracies']
            
            print('>>checkpoint loaded!')
        print("\n====>Global model training started...\n")

        folder_name =f"stats/{dd_type}/{self.dataset_name}/{rule}/{attackers_ratio}"
        
        #for p in trust_w:
        #    print(self.peers[p].peer_id, self.peers[p].peer_type)
        for peer in self.choose_peers():
                if self.peers[peer].peer_type=="attacker":
                    malicious_nodes.append(self.peers[peer].peer_id)


 
        if rule=="localEval":
            print_to_file(f"malcious_nodes = {malicious_nodes}", f"{folder_name}/malicous_nodes.txt")

        saved_global_accuracies = {}
        aggregation_time = []
        average_loss = []
        for epoch in tqdm_notebook(range(start_round, self.global_rounds)):
            gc.collect()
            torch.cuda.empty_cache()
            
            # if epoch % 20 == 0:
            #     clear_output()  
            print(f'\n | Global training round : {epoch+1}/{self.global_rounds} |\n')
            selected_peers = self.choose_peers()


            local_weights, local_grads, local_models, lossGlobal, performed_attacks = [], [], [], [], []  
            peers_types , current_trust , peerAccuracy = [], [], []

            
            number_of_peers = selected_peers.size
     
            i = 1        
            attacks = 0
            Peer._performed_attacks = 0
            total_accuracy = 0.0
            saved_accuracy = {}
            normalize(trust_w, bad_nodes)
            for peer in selected_peers:
                peers_types.append(mapping[self.peers[peer].peer_type])
                # print(i)
                # print('\n{}: {} Starts training in global round:{} |'.format(i, (self.peers_pseudonyms[peer]), (epoch + 1)))

                # peer_update = the state_dict form of the peer model
                # peer_grad = gradients i.e weights and biases

                peer_update, peer_grad, peer_local_model, peer_loss, attacked, t ,  peer_accuracy  = self.peers[peer].participant_update(epoch, copy.deepcopy(simulation_model), 
                attack_type = attack_type, malicious_behavior_rate = malicious_behavior_rate, 
                source_class = source_class, target_class = target_class, dataset_name = self.dataset_name)

                


                current_peer = self.peers[peer].peer_id
                i+= 1
                attacks+= attacked
                if(rule == "localEval" and current_peer in bad_nodes):
                    continue

                local_weights.append(peer_update)
                local_grads.append(peer_grad)
                peerAccuracy.append((current_peer,peer_accuracy))
                saved_accuracy[current_peer] = peer_accuracy 

                total_accuracy += peer_accuracy
                local_models.append( peer_local_model) 
                current_trust.append(trust_w[current_peer])

            avg_accuracy = total_accuracy / len(local_weights)
            alphaAntiFlipper = 0.09
            if rule=="localEval" and epoch >= 0:
                for peerID , peer_accuracy in peerAccuracy:
                    # hisotircal consideration
                    # trust_w[peerID] = (1.0 - alphaAntiFlipper) * trust_w[peerID] + alphaAntiFlipper * (1.0 * peer_accuracy / total_accuracy)

                    ## This is the new quadratic implementation for trust updates
                    diff = peer_accuracy - avg_accuracy
                    adjustment = 0.1 * (diff * diff) 
                    if diff < 0:
                        adjustment *= -1
                    trust_w[peerID] = trust_w[peerID] + adjustment
                    
                    if trust_w[peerID] < 0:
                        trust_w[peerID] = 0
                    if trust_w[peerID] < 0.0005 and epoch > 1:
                         bad_cnt[peerID] += 1
                    if bad_cnt[peerID] > 5 and  peerID not in bad_nodes:
                        bad_nodes.add(peerID)
                        print_to_file(f"{peerID} got detected at round {epoch}", f"{folder_name}/bad_nodes.txt")


                normalize(trust_w, bad_nodes)
                print_to_file(f"trust[{epoch}] =  {trust_w}", f"{folder_name}/trust.txt")

                #print(f"local accuracies : {local_losses}")    
                #print(f"peer  : {peerAccuracy}")
                #Peer Accuracies Per Round
                print_to_file(f"peer_accuracies[{epoch}] = {saved_accuracy}", f"{folder_name}/accuracies.txt")
            
            total_trust_value = 0.0
            for value in trust_w.values():
                total_trust_value += value
                
            print_to_file(f"trust_sum : {total_trust_value}", "/home/ndag/Abid/AntiFlipper/stats/marks.txt")
            scores = np.zeros(len(local_weights))
            # Expected malicious peers
            f = int(self.num_peers*self.attackers_ratio)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
            if rule == 'median':
                    cur_time = time.time()
                    global_weights = simple_median(local_weights)
                    cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'localEval':
                    cur_time = time.time()
                    if len(local_weights) > 0 : 
                        global_weights = average_weights(local_weights, copy.deepcopy(current_trust))
                    cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'rmedian':
                cur_time = time.time()
                global_weights = Repeated_Median_Shard(local_weights)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'tmean':
                    cur_time = time.time()
                    trim_ratio = self.attackers_ratio*self.num_peers/len(selected_peers)
                    global_weights = trimmed_mean(local_weights, trim_ratio = trim_ratio)
                    cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'mkrum':
                cur_time = time.time()
                goog_updates = Krum(local_models, f = f, multi=True)
                scores[goog_updates] = 1
                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'foolsgold':
                cur_time = time.time()
                scores = fg.score_gradients(local_grads, selected_peers)
                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time + t)

            elif rule == 'Tolpegin':
                cur_time = time.time()
                scores = tolpegin.score(copy.deepcopy(self.global_model), 
                                            copy.deepcopy(local_models), 
                                            peers_types = peers_types,
                                            selected_peers = selected_peers)
                global_weights = average_weights(local_weights, scores)
                t = time.time() - cur_time
                print('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)
            
            elif rule == 'FLAME':
                cur_time = time.time()
                global_weights = FLAME(copy.deepcopy(self.global_model).cpu(), copy.deepcopy(local_models), noise_scalar)
                t = time.time() - cur_time
                print('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)


            elif rule == 'lfighter':
                cur_time = time.time()
                global_weights = lfd.aggregate(copy.deepcopy(simulation_model), copy.deepcopy(local_models), peers_types)
                cpu_runtimes.append(time.time() - cur_time)

            
            elif rule == 'fedavg':
                cur_time = time.time()
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                cpu_runtimes.append(time.time() - cur_time)
            
            else:
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                ##############################################################################################
            

            
            #Plot honest vs attackers
            # if attack_type == 'label_flipping' and epoch >= 10 and epoch < 20:
            #     plot_updates_components(local_models, peers_types, epoch=epoch+1)   
            #     plot_layer_components(local_models, peers_types, epoch=epoch+1, layer = 'linear_weight')  
            #     plot_source_target(local_models, peers_types, epoch=epoch+1, classes= [source_class, target_class])
            # update global weights
           
            end_event.record()
            torch.cuda.synchronize()
            time_elapsed = start_event.elapsed_time(end_event)
            aggregation_time.append(time_elapsed)
            g_model = copy.deepcopy(simulation_model)
            simulation_model.load_state_dict(global_weights)     

            if epoch >= self.global_rounds-10:
                last10_updates.append(global_weights) 

            current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
            saved_global_accuracies[epoch] = current_accuracy
            average_loss.append(test_loss)

            if np.isnan(test_loss):
                simulation_model = copy.deepcopy(g_model)
                noise_scalar = noise_scalar*0.5
            
            global_accuracies.append(np.round(current_accuracy, 2))
            test_losses.append(np.round(test_loss, 4))
            performed_attacks.append(attacks) 


            state = {
                'epoch': epoch,
                'state_dict': simulation_model.state_dict(),
                'global_model':g_model,
                'local_models':copy.deepcopy(local_models),
                'last10_updates':last10_updates,
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies
                }
            savepath = './checkpoints/'+ self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7'


            torch.save(state,savepath)
            del local_models
            del local_weights
            del local_grads
            gc.collect()
            torch.cuda.empty_cache()



            # print("***********************************************************************************")
            #print and show confusion matrix after each global round
            actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
            classes = list(self.labels_dict.keys())
            print('{0:10s} - {1}'.format('Class','Accuracy'))
           
           
            #generating confusing matxi
            for i, r in enumerate(confusion_matrix(actuals, predictions)):
                print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
                if i == source_class:
                    source_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))


            # basically i==n-1
            if epoch == self.global_rounds-1:


                #printing only last 10 updates to give some idea about training
                print('Last 10 updates results')
                global_weights = average_weights(last10_updates, 
                np.ones([len(last10_updates)]))
                simulation_model.load_state_dict(global_weights) 
                current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
                global_accuracies.append(np.round(current_accuracy, 2))
                test_losses.append(np.round(test_loss, 4))
                performed_attacks.append(attacks)
                print("***********************************************************************************")




                #print and show confusion matrix after each global round
                actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
                classes = list(self.labels_dict.keys())
                print('{0:10s} - {1}'.format('Class','Accuracy'))
                asr = 0.0
                for i, r in enumerate(confusion_matrix(actuals, predictions)):
                    print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
                    if i == source_class:
                        source_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))
                        asr = np.round(r[target_class]/np.sum(r)*100, 2)

        state = {
                'state_dict': simulation_model.state_dict(),
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies,
                'asr':asr,
                'avg_cpu_runtime':np.mean(cpu_runtimes)
                }
        savepath = './results/'+ self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7'
        print_to_file(f"global_accurices = {saved_global_accuracies}", f"{folder_name}/global_accuracies.txt")
        if rule=="localEval":
            print_to_file(f"detected_nodes = {bad_nodes}", f"{folder_name}/detected_nodes.txt")
        print_to_file(f"average aggregation time : = {aggregation_time}", f"{folder_name}/aggregation_time.txt")
        print_to_file(f"Test Loss : = {average_loss}", f"{folder_name}/test_loss.txt")

            
        torch.save(state,savepath)            
        print('Global accuracies: ', global_accuracies)
        print('Class {} accuracies: '.format(source_class), source_class_accuracies)
        print(f"trust Weightes : {trust_w}")
        print('Test loss:', test_losses)
        print('Attack succes rate:', asr)
        print('Average CPU aggregation runtime:', np.mean(cpu_runtimes))
