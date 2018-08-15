import dynet as dy
from math import log
import random
import pickle




train_file='old-church-slavonic-train-low'
validation_file='old-church-slavonic-dev'
num_epochs = 300
model_file = 'initial.model'



#Make a character set
EOS = "<EOS>"
character_set = set([EOS])

train_raw_data = [l.strip().split('\t') for l in open(train_file).read().split('\n') if l.strip() != '']
train_features = [[c for c in lemma] +[c for c in wf] for lemma, wf,_ in train_raw_data]
train_tags = [ tags.split(';') for _, _, tags in train_raw_data]    
val_data = [l.strip().split('\t') for l in open(validation_file).read().split('\n') if l.strip() != '']
val_features = [[c for c in lemma] + [c for c in wf] for lemma, wf, _ in val_data]
val_tags = [tags.split(';') for _, _, tags in val_data]

for wf in train_features + train_tags + val_features + val_tags:
        for c in wf:
            character_set.add(c)

#Extract the training data
#train_data = []
#train_raw_data = open(train_file).read()
#for line in train_raw_data.split('\n'):
#    if line.strip() != '':
#        train_data.append(line.strip().split('\t'))
#Make character lists of the features and tags
#train_features = []
#train_tags = []
#for lemma,target,_ in train_data:
#    train_features.append([a for a in lemma]+ [a for a in target])
#for _,_,tags in train_data:
#    train_tags.append(tags.split(';'))
#Add charaters to character set
#for word in train_features+train_tags:
#    for character in word:
#        character_set.add(character)
    

#Extract the validation data
#val_data = []
#val_raw_data = open(validation_file).read()
#for line in val_raw_data.split('\n'):
#    if line.strip() != '':
#        val_data.append(line.strip().split('\t'))
#Make character lists of the features and tags
#val_features = []
#val_tags = []
#for lemma,target,_ in val_data:
#    val_features.append([a for a in lemma]+ [a for a in target])
#for _,_,tags in train_data:
#    val_tags.append(tags.split(';'))
#Add charaters to character set
#for word in val_features+val_tags:
#    for character in word:
#        character_set.add(character)

class LSTM():

    def __init__(self, LAYERS = 2, INPUT_DIM = 32, HIDDEN_DIM = 32, attention_size = 32, EPOCHS = 1, character_set = None):
        
        #Define the LSTM parameters
        self.LAYERS = LAYERS
        self.INPUT_DIM = INPUT_DIM
        self.HIDDEN_DIM = HIDDEN_DIM
        self.attention_size = attention_size
        self.EPOCHS = EPOCHS

        #Intialize the LSTM model
        self.characters = character_set
        self.int2char = sorted(list(self.characters))
        self.char2int = {c:i for i,c in enumerate(self.int2char)}

        self.VOCAB_SIZE = len(self.characters)
            
        self.model = dy.Model()

        self.enc_fwd_lstm = dy.LSTMBuilder(self.LAYERS, self.INPUT_DIM, self.HIDDEN_DIM, self.model)
        self.enc_bwd_lstm = dy.LSTMBuilder(self.LAYERS, self.INPUT_DIM, self.HIDDEN_DIM, self.model)
        self.dec_lstm = dy.LSTMBuilder(self.LAYERS, self.HIDDEN_DIM*2+self.INPUT_DIM, self.HIDDEN_DIM, self.model)
            
        self.input_lookup = self.model.add_lookup_parameters((self.VOCAB_SIZE, self.INPUT_DIM))
        self.attention_w1 = self.model.add_parameters( (self.attention_size, self.HIDDEN_DIM*2))
        self.attention_w2 = self.model.add_parameters( (self.attention_size, self.HIDDEN_DIM*self.LAYERS*2))
        self.attention_v = self.model.add_parameters( (1, self.attention_size))
        self.decoder_w = self.model.add_parameters( (self.VOCAB_SIZE, self.HIDDEN_DIM))
        self.decoder_b = self.model.add_parameters( (self.VOCAB_SIZE))
        self.output_lookup = self.model.add_lookup_parameters((self.VOCAB_SIZE, self.INPUT_DIM))

        print('Initialization Successful')
    def embed_sentence(self, sentence):
        sentence = [EOS] + list(sentence) + [EOS]
        # Skip unknown self.characters.
        sentence = [self.char2int[c] for c in sentence if c in self.char2int]
        return [self.input_lookup[char] for char in sentence]


    def run_lstm(self, init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors


    def encode_sentence(self, enc_fwd_lstm, enc_bwd_lstm, sentence):
        sentence_rev = list(reversed(sentence))

        fwd_vectors = self.run_lstm(enc_fwd_lstm.initial_state(), sentence)
        bwd_vectors = self.run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

        return vectors


    def attend(self, input_mat, state, w1dt):
        w2 = dy.parameter(self.attention_w2)
        v = dy.parameter(self.attention_v)
        w2dt = w2*dy.concatenate(list(state.s()))
        att_weights = dy.softmax(dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt))))
        context = input_mat * att_weights
        return context


    def decode(self, dec_lstm, vectors, output):
        output = [EOS] + list(output) + [EOS]
        output = [self.char2int[c] for c in output]

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)
        w1 = dy.parameter(self.attention_w1)
        input_mat = dy.concatenate_cols(vectors)
        w1dt = None

        last_output_embeddings = self.output_lookup[self.char2int[EOS]]
        s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.HIDDEN_DIM*2), last_output_embeddings]))
        loss = []

        for char in output:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = w * s.output() + b
            probs = dy.softmax(out_vector)
            last_output_embeddings = self.output_lookup[char]
            loss.append(-dy.log(dy.pick(probs, char)))
        loss = dy.esum(loss)
        return loss


    def generate(self, in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
        embedded = self.embed_sentence(in_seq)
        encoded = self.encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)
        w1 = dy.parameter(self.attention_w1)
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = self.output_lookup[self.char2int[EOS]]
        s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.HIDDEN_DIM * 2), last_output_embeddings]))

        out = ''
        count_EOS = 0
        for i in range(len(in_seq)*2):
            if count_EOS == 2: break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = w * s.output() + b
            probs = dy.softmax(out_vector).vec_value()
            next_char = probs.index(max(probs))
            last_output_embeddings = self.output_lookup[next_char]
            if self.int2char[next_char] == EOS:
                count_EOS += 1
                continue

            out += self.int2char[next_char]
        return out

    def get_loss(self, input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
        dy.renew_cg()
        embedded = self.embed_sentence(input_sentence)
        encoded = self.encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)
        return self.decode(dec_lstm, encoded, output_sentence)


    def train(self, isentences,osentences, idevsentences,odevsentences, ofilen):
        trainer = dy.AdamTrainer(self.model, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8)
        iopairs = list(zip(isentences,osentences))
        random.shuffle(iopairs)
        best_dev_acc = 0
        for i in range(self.EPOCHS):
            loss_value = 0
    #        random.shuffle(iopairs)
            for isentence, osentence in iopairs:
                loss = self.get_loss(isentence, osentence, self.enc_fwd_lstm, self.enc_bwd_lstm, self.dec_lstm)
                loss_value += loss.value()
                loss.backward()
                trainer.update()

            corr = 0
            for ip,op in zip(idevsentences,odevsentences):
                dy.renew_cg()
                sys_o = self.generate(ip, self.enc_fwd_lstm, self.enc_bwd_lstm, self.dec_lstm)
                if ''.join(op) == sys_o:
                    corr += 1
            dev_acc = corr * 100.0 / len(idevsentences)

            print("EPOCH %u: LOSS %.2f, DEV ACC %.2f" % (i+1, loss_value/len(iopairs), dev_acc))
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                print("SAVING!")
                self.save_model(ofilen)

    def save_model(self, ofilen):
        self.model.save(ofilen)    
        pickle.dump((self.characters,self.int2char,self.char2int),open("%s.chars.pkl" % ofilen,"wb"))

    def load_model(self, ifilen):
        self.model.load(ifilen)

    def test(self, itestsentences, ofile):
        for ilemma,ilabels in itestsentences:
            dy.renew_cg()
            sys_o = generate(ilemma+ilabels, self.enc_fwd_lstm, self.enc_bwd_lstm, self.dec_lstm)
            ofile.write("%s\t%s\t%s\n" % (''.join(ilemma),sys_o,';'.join(ilabels)))



print('blabla')
train_lstm = LSTM(EPOCHS = num_epochs, character_set = character_set)
train_lstm.train(train_features,train_tags,val_features,val_tags, model_file)


