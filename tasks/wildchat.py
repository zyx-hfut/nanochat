
from datasets import load_dataset
from tasks.common import Task

class WildChat(Task):
    """ everyday conversation dataset. train is 2.26K rows, test is 0.119K rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["Chinese", "English"], "WildChat split must be Chinese|English"
        self.ds = load_dataset("/home/featurize/data/wildchat",split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["conversation"]

        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:] # optional system message is OK
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, "Wildchat Conversation No think messages must have at least 2 messages"
        for i, message in enumerate(rest_messages):
            # user and assistant alternate as user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"
        # ---------------------------------------------------------------------
        # create and return the Conversation object (ok to emit the system message too)
        conversation = {
            "messages": messages,
        }
        return conversation
