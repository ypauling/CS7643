import os
import utils
import nltk
import pickle
from params import get_parser
from proc import detect_ingrs

parser = get_parser()
params = parser.parse_args()
create = params.create_bigrams

print('Loading dataset...')
DATASET = params.dataset
dataset = utils.Layer.merge(
    [utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS], DATASET)
print('Finishi loading...')

if create:
    print('Creating bigrams...')
    titles = []

    for entry in dataset:
        title = entry['title']
        if entry['partition'] == 'train':
            titles.append(title)

    output_folder = params.bigram_folder
    suffix = params.suffix
    filename = os.path.join(output_folder, 'titles{}.txt'.format(suffix))
    with open(filename, 'w') as f:
        for t in titles:
            f.write(t + " ")

    from nltk.corpus import stopwords
    with open(filename, 'r') as f:
        raw = f.read()
        tokens = nltk.word_tokenize(raw)
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if w not in stopwords.words('english')]
        bgs = nltk.bigrams(tokens)
        fdist = nltk.FreqDist(bgs)
        pickle.dump(fdist, open(os.path.join(
            output_folder, 'bigrams{}.pkl'.format(suffix)), 'wb'))

else:
    N = 2000
    MAX_CLASSES = 1000
    MIN_SAMPLES = params.tsamples
    n_class = 1
    ind2class = {}
    class_dict = {}

    fbd_chars = [
        ",", "&", "(", ")", "'", "'s", "!", "?",
        "%", "*", ".", "free", "slow", "low",
        "old", "easy", "super", "best", "-",
        "fresh", "ever", "fast", "quick", "fat",
        "ww", "n'", "'n", "n", "make", "con",
        "e", "minute", "minutes", "portabella",
        "de", "of", "chef", "lo", "rachael",
        "poor", "man", "ii", "i", "year", "new", "style"]

    print('Loading ingredients vocabulary...')
    with open(params.vocab, 'r') as f:
        ingr_vocab = {w.rstrip(): i+2 for i, w in enumerate(f)}
        ingr_vocab['</i>'] = 1

    ningrs_list = []
    for i, entry in enumerate(dataset):
        ingr_detected = detect_ingrs(entry, ingr_vocab)
        ningrs_list.append(len(ingr_detected))

    output_folder = params.bigram_folder
    suffix = params.suffix
    fdist = pickle.load(
        open(os.path.join(output_folder, 'bigrams{}.pkl'.format(suffix)),
             'rb'))
    Ntop = fdist.most_common(N)

    queries = []
    for item in Ntop:
        bg = item[0]
        counts = {'train': 0, 'val': 0, 'test': 0}

        if bg[0] in fbd_chars or bg[1] in fbd_chars:
            continue

        query = '{} {}'.format(bg[0], bg[1])
        queries.append(query)
        matching_ids = []

        for i, entry in enumerate(dataset):
            ninstrs = len(entry['instructions'])
            imgs = entry.get('images')
            ningrs = ningrs_list[i]
            title = entry['title'].lower()
            eid = entry['id']

            if query in title and ninstrs < params.maxlen and imgs and \
                    ningrs < params.maxlen and ningrs != 0:
                if eid not in class_dict or class_dict[eid] == 0:
                    class_dict[eid] = n_class
                    counts[entry['partition']] += 1
                    matching_ids.append(eid)
            else:
                if eid not in class_dict:
                    class_dict[eid] = 0

        if counts['train'] > MIN_SAMPLES and counts['val'] > 0 and \
                counts['test'] > 0:
            ind2class[n_class] = query
            print(n_class, query, counts)
            n_class += 1
        else:
            for eid in matching_ids:
                class_dict[eid] = 0

        if n_class > MAX_CLASSES:
            break

    food101 = []
    with open(params.f101_cats, 'r') as f101:
        for item in f101:
            current_class = item.lower().rstrip().replace('_', ' ')
            if current_class not in queries:
                food101.append(current_class)

    for query in food101:
        counts = {'train': 0, 'val': 0, 'test': 0}
        matching_ids = []

        for i, entry in enumerate(dataset):
            ninstrs = len(entry['instructions'])
            imgs = entry.get('images')
            ningrs = ningrs_list[i]
            title = entry['title'].lower()
            eid = entry['id']

            if query in title and ninstrs < params.maxlen and imgs and \
                    ningrs < params.maxlen and ningrs != 0:
                if eid not in class_dict or class_dict[eid] == 0:
                    class_dict[eid] = n_class
                    counts[entry['partition']] += 1
                    matching_ids.append(eid)

            else:
                if eid not in class_dict:
                    class_dict[eid] = 0

        if counts['train'] > MIN_SAMPLES and counts['val'] > 0 and \
                counts['test'] > 0:
            ind2class[n_class] = query
            print(n_class, query, counts)
            n_class += 1
        else:
            for eid in matching_ids:
                class_dict[eid] = 0

    ind2class[0] = 'background'
    print(len(ind2class))

    with open(os.path.join(output_folder, 'classes{}.pkl'.format(suffix)),
              'wb') as f:
        pickle.dump(class_dict, f)
        pickle.dump(ind2class, f)
