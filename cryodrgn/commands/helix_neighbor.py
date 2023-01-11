import argparse
import pandas as pd
import numpy as np
import time

start = time.process_time()
def add_args(parser):
    parser.add_argument('star', help='Input starfile' )
    parser.add_argument('--relion31', action='store_true',help='whether use relion31 files or not')
    parser.add_argument('--filament', action='store_true', help='take the whole filament as neighbor')
    parser.add_argument('--w', help='number of filament as neighbor')
    parser.add_argument('--o', help='output filament id')
    return parser

def star2dataframe(filename, relion31=None):
    Rvar = []  # read the variables metadata
    Rdata = []  # read the data
    start_read_line=1
    if relion31:
        start_read_line=20
    for star_line in open(filename).readlines()[start_read_line:]:
        if star_line.find("_rln") != -1:
            var = star_line.split()
            Rvar.append(var[0])
        #    Rvar_len = Rvar_len+1
        elif star_line.find("data_") != -1 or star_line.find("loop_") != -1 or len(star_line.strip()) == 0:
            continue
        else:
            Rdata.append(star_line.split())

    print(Rdata[0],Rvar)

    data = pd.DataFrame(data=Rdata,columns=Rvar)

    assert ("_rlnImageName" in data)
    tmp = data["_rlnImageName"].str.split("@", expand=True)
    indices, filenames = tmp.iloc[:, 0], tmp.iloc[:, -1]
    indices = indices.astype(int) - 1
    data["pid"] = indices
    data["filename"] = filenames

    if "_rlnClassNumber" in data:
        data.loc[:, "class"] = data["_rlnClassNumber"]
    if "_rlnHelicalTubeID" in data:
        data.loc[:, "helicaltube"] = data["_rlnHelicalTubeID"].astype(int) - 1
    if "_rlnAnglePsiPrior" in data:
        data.loc[:, "phi0"] = data["_rlnAnglePsiPrior"].astype(float).round(3) - 90.0
    return data

def extract_helical_select(dataframe):
    filament_data = dataframe.groupby(['filename', 'helicaltube'])
    filament_index = list(filament_data.groups.keys())
    helicaldic = {}
    helicalnum = []
    dtype = [('class2D', int), ('place', int), ('index', int)]
    for i in range(len(filament_index)):
        name = '-'.join(map(str, filament_index[i]))
        helicaldic[name] = []
        helicalnum = helicalnum + [name]
    print('The filament number are: ', len(helicalnum))
    print('The number of particles are:', len(dataframe))
    for i in range(len(dataframe)):
        particle = dataframe.iloc[i]
        ID = str(particle['filename']) + '-' + str(particle['helicaltube'])
        helicaldic[ID] = helicaldic[ID] + [(particle['class'], particle['pid'], i)]
        if i % 10000 == 0:
            end = time.process_time()
            elapsed_time = (end - start)/60
            print(i, '%s mins' % elapsed_time)
    for i in range(len(helicalnum)):
        lst = np.array(helicaldic[helicalnum[i]], dtype=dtype)
        helicaldic[helicalnum[i]] = np.sort(lst, order='place')
    print('finish converting')
    for i in range(10):
        print(helicaldic[helicalnum[i]])
    corpus = list(helicaldic.values())

    corpus_ignore = []
    for i in range(len(corpus)):
        corpus_row = []
        lst = corpus[i]
        count = lst[0][1]
        for j in range(len(lst)):
            particle = lst[j]
            if count == int(particle[1]):
                corpus_row.append(particle[2])
                count += 1
            else:
                while 1:
                    if count == int(lst[j][1]):
                        corpus_row.append(particle[2])
                        count += 1
                        break
                    corpus_row += ['?']
                    count += 1
        corpus_ignore.append(corpus_row)

    return corpus_ignore

def create_pairs(corpus_ignore,w=2):
    w = int(w)
    context_tuple_list = []
    count=0
    for text in corpus_ignore:
        for i, word in enumerate(text):
            if word == '?':
                continue
            first_context_word_index = max(0, i - w)
            last_context_word_index = min(i + w + 1, len(text))
            lst=[]
            for j in range(first_context_word_index, last_context_word_index):
                neighbor = text[j]
                if neighbor == '?':
                    continue
                lst.append(neighbor)
                count+=1
            context_tuple_list.append([word,lst])
    print("There are {} pairs of target and context words".format(count))
    context_tuple_list=np.array(context_tuple_list)
    ind=np.argsort(context_tuple_list[:,0])
    context_tuple_list=context_tuple_list[ind]
    context_tuple_list=context_tuple_list[:,-1]
    return context_tuple_list


def get_pair_index(dataframe, w=None, filament=None):
    filament_pairs=[]
    if filament:
        for i in range(len(dataframe)):
            if i%10000==0:
                end = time.process_time()
                elapsed_time = end - start
                print(f"Elapsed time: {elapsed_time} seconds for {i} particle")
            tube_id=list(dataframe['helicaltube'])[i]
            mic_id=list(dataframe['filename'])[i]
            #print(tube_id,mic_id)
            lst=dataframe[(dataframe['helicaltube']==tube_id)&(dataframe['filename']==mic_id)].index.to_numpy()
            filament_pairs.append(lst)
    else:
        window_size=w
        corpus_ignore=extract_helical_select(dataframe)
        filament_pairs=create_pairs(corpus_ignore, window_size)

    filament_pairs=np.array(filament_pairs)
    print(np.shape(filament_pairs))
    print(filament_pairs[0])
    return filament_pairs



def main(args):

    dataframe=star2dataframe(args.star, relion31=args.relion31)
    neighbor_id=get_pair_index(dataframe, w=args.w, filament=args.filament)
    np.save(args.o, neighbor_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
