class State:
    def __init__(self, sys_name, usr_name):
        self.sys = sys_name
        self.usr = usr_name
        self.history = []

    def __iter__(self):
        return iter(self.history)

    def __len__(self):
        return len(self.history) // 2

    def __getitem__(self, index):
        return self.history[index]

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, State):
            return False
        return self.history == value.history

    def copy(self):
        new_state = State(self.sys, self.usr)
        new_state.from_history(self.history.copy())
        return new_state

    def from_history(self, history):
        self.history = history
        return self

    def add(self, role, action, content):
        if len(self.history) % 2 == 0:
            assert role == self.sys
        else:
            assert role == self.usr
        self.history.append((role, action, content))

    def to_string_represent(self, keep_sys_action=False, keep_usr_action=False, max_turn_to_display=-1):
        history = []
        num_turns_to_truncate = 0
        if max_turn_to_display > 0:
            num_turns_to_truncate = max(0, len(self.history) // 2 - max_turn_to_display)
        for i, (role, action, content) in enumerate(self.history):
            if (i // 2) < num_turns_to_truncate:
                continue
            if i % 2 == 0:
                assert role == self.sys
                if keep_sys_action:
                    history.append(f"{role}: [{action}] {content}".strip())
                else:
                    history.append(f"{role}: {content}".strip())
            else:
                assert role == self.usr
                if keep_usr_action:
                    history.append(f"{role}: [{action}] {content}".strip())
                else:
                    history.append(f"{role}: {content}".strip())
        return "\n".join(history).strip()

    def get_turn_utterance(self, turn, role):
        if role == self.sys:
            return self.history[turn * 2][-1]
        else:
            return self.history[turn * 2 + 1][-1]
