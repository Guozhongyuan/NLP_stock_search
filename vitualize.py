import numpy as np
from pathlib import Path


import numpy as np
from tqdm import tqdm
data = np.load('./data_ignore/vectors_gpt2.pkl', allow_pickle=True)
ticker_names = []
ticker_ids = []
vectors = []
for idx in tqdm(range(len(data))):
    if 'vector' in data[idx].keys():
        vectors.append(data[idx]['vector'])
        ticker_ids.append(data[idx]['ticker_id'])
        ticker_names.append(data[idx]['ticker_name'])
vectors = np.stack(vectors)


words = ticker_ids
logdir = Path('projector/')  # 文件存储目录

metadata_filename = 'metadata.tsv'
lines = ["ticker_id\tticker_name"]  # 此处我们存成两列，你也可以不要表头，只存词

# for i, word in enumerate(words):
#     lines.append(f"{i}\t{word}")

for id, name in zip(ticker_ids, ticker_names):
    lines.append(f"{id}\t{name}")

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