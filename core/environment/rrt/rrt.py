class TreeNode:
    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

    def retrace(self):
        node = self
        sequence = []
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def __str__(self):
        return "TreeNode(" + str(self.config) + ")"

    __repr__ = __str__


def configs(nodes):
    if nodes is None:
        return None
    return list(map(lambda node: node.config, nodes))
