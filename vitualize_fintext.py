import numpy as np
from pathlib import Path


import numpy as np
from tqdm import tqdm
data = np.load('./data_ignore/vectors_fintext.pkl', allow_pickle=True)
words = []
vectors = []
for idx in tqdm(range(len(data))):
    if 'vector' in data[idx].keys():
        vectors.append(data[idx]['vector'])
        words.append(data[idx]['ticker'])
vectors = np.stack(vectors)


logdir = Path('projector/')  # 文件存储目录

metadata_filename = 'metadata.tsv'
lines = ["name\tkeywords"]


for word in words:
    lines.append(f"{word}")

logdir.joinpath(metadata_filename).write_text("\n".join(lines), encoding="utf8")

tensor_filename = 'tensor.tsv'
lines = ["\t".join(map(str, vector)) for vector in vectors]
logdir.joinpath(tensor_filename).write_text("\n".join(lines), encoding="utf8")



'''
    config.pbtxt, projector need it to show
'''
from tensorboard.plugins import projector
from pathlib import Path


metadata_filename = 'metadata.tsv'
tensor_filename = 'tensor.tsv'
logdir = Path('projector/')

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.metadata_path = metadata_filename
embedding.tensor_path = tensor_filename
projector.visualize_embeddings(logdir, config)

# tensorboard --logdir=projector/