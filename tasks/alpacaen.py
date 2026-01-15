"""
CN i wonder why Dataset.
Aggregates multiple Chinese SFT subsets.
Converts prompt/response format to standard messages format.
Supports stacking independent pairs to simulate multi-turn dialogues.
"""

from datasets import load_dataset, concatenate_datasets
from tasks.common import Task

class Alpaca(Task):
    """
    English Alpaca SFT dataset.
    Original columns: 'instruction', 'input', 'output' (Strings)
    Target columns: 'messages' (List of Dicts)
    """

    def __init__(self, split, stack_turns=1, seed=42, **kwargs):
        """
        Args:
            split: "train" or "test"
            stack_turns: int. 
                If 1, returns standard single-turn. 
                If > 1, merges N consecutive independent QA pairs into one multi-turn dialogue.
                (Helps model learn role switching, even if context is disjoint).
            seed: random seed for shuffling and splitting.
        """
        super().__init__(**kwargs)
        self.split = split
        self.stack_turns = max(1, int(stack_turns))
        
        path = "/home/featurize/data/alpaca_en"
        
        # 1. 加载数据
        self.ds = load_dataset(path, split="train")
        # 2. 划分 Train / Test (因为原始只有 train)
        # 我们按照 95% 训练, 5% 测试进行切分
        self.ds = self.ds.shuffle(seed=seed)
        # 使用 huggingface 的 train_test_split 功能
        ds_split = self.ds.train_test_split(test_size=0.05, seed=seed)
        
        if split == "train":
            self.ds = ds_split["train"]
        else:
            self.ds = ds_split["test"]
            
        self.length = len(self.ds)

    def num_examples(self):
        # 修改点 1：总样本数除以 stack_turns（向下取整）
        # 比如有 100 条数据，每 3 条拼成 1 个，那我们就只有 33 个训练样本
        if self.stack_turns > 1:
            return self.length // self.stack_turns
        return self.length

    def get_example(self, index):
        messages = []
        
        # 修改点 2：计算起始索引，实现“跳跃”
        # 如果 index=0, start=0; index=1, start=3; index=2, start=6...
        start_idx = index * self.stack_turns
        
        for i in range(self.stack_turns):
            # 获取绝对索引
            curr_idx = start_idx + i
            
            # 安全检查（理论上 num_examples 已经限制了，但加个保险）
            if curr_idx >= self.length:
                break
                
            row = self.ds[curr_idx]
            instruction = row.get("instruction", "")
            input = row.get("input", "")
            output = row.get("output", "")

            # 清洗
            if not isinstance(instruction, str): instruction = str(instruction)
            if not isinstance(input, str): input = str(input)
            if not isinstance(output, str): output = str(output)

            messages.append({"role": "user", "content": instruction+"\n"+input})
            messages.append({"role": "assistant", "content": output})

        conversation = {
            "messages": messages,
        }
        return conversation