
### files
- `query_gpt2_vectors.ipynb` 保存gpt2的30000维度向量
- `query_tencent_vectors.ipynb`  保存腾讯api的300维度向量
- `query_keywords.ipynb`  保存词袋模型向量

---

### vistualization
参考：
https://alanlee.fun/2021/12/17/tensorboard-embedding-projector/
https://branyang.gitbooks.io/tfdocs/content/get_started/embedding_viz.html
- `vitualize.py`
- `tensorboard --logdir=projector/`

---

### TODO
1. jieba添加公司名再处理语料，预期能通过直接搜索公司名给出结果
2. 只finetune全连接层会怎样，用高级语义查询测试
3. finetune1个epoch就够了？不然会过拟合，前端查询不到
4. 查询语句要长一些，仅一个单词和重复它的结果不一样