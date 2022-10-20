import argparse
import pandas as pd
import numpy as np

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

def get_pair_index(dataframe, w=None, filament=None):
    filament_pairs=[]
    if filament:
        for i in range(len(dataframe)):
            tube_id=list(dataframe['helicaltube'])[i]
            mic_id=list(dataframe['filename'])[i]
            lst=dataframe[(dataframe['helicaltube']==tube_id)&(dataframe['filename']==mic_id)].index.to_numpy()
            filament_pairs.append(lst)
    filament_pairs=np.array(filament_pairs)
    print(np.shape(filament_pairs),filament_pairs[0])
    return filament_pairs



def main(args):

    dataframe=star2dataframe(args.star, relion31=args.relion31)
    neighbor_id=get_pair_index(dataframe, w=args.w, filament=args.filament)
    np.save(args.o, neighbor_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())