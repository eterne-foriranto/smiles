#%%
from matplotlib import pyplot as plt
from networkx import Graph, compose, disjoint_union, draw, union
from pysmiles import write_smiles
from rdkit.Avalon.pyAvalonTools import GetCanonSmiles
from rdkit import Chem
#%%


class SymDict:
    def __init__(self):
        self.inner_dict = {}

    def get(self, key):
        frozen_key = frozenset(key)
        if frozen_key in self.inner_dict:
            return self.inner_dict[frozen_key]

    def set(self, key, value):
        self.inner_dict[frozenset(key)] = value
#%%
m = Chem.MolFromMolFile('pyrrole.mol')
#%%
def get_pyrrolle_g():
    nitrogen_of_pyrrole = 'N', {'element': 'N', 'hcount': 1}
    carbons_of_pyrrole = [(f'C{i}', {'element': 'C', 'hcount':
        1}) for i in range(1, 5)]

    pyrrole_list = [nitrogen_of_pyrrole, *carbons_of_pyrrole]
    g = Graph()

    g.add_nodes_from(pyrrole_list)

    g.add_edge('N', 'C1', order=1)
    g.add_edge('C2', 'C1', order=2)
    g.add_edge('C2', 'C3', order=1)
    g.add_edge('C4', 'C3', order=2)
    g.add_edge('N', 'C4', order=1)

    # g.nodes['C4']['synth'] = 'c4'
    # g.nodes['C1']['synth'] = 'c1'

    return g

#%%
def get_formaldehyde():
    f = Graph()
    f.add_nodes_from([
        ('c', {'element': 'C', 'hcount': 2}),
        ('o', {'element': 'O'})
    ])
    f.add_edge('c', 'o', order=2)
    # f.nodes['c']['synth'] = 'c'

    return f
#%%
def get_node_by_synth_label(g, label):
    nodes_view = g.nodes
    for node in nodes_view:
        if nodes_view[node].get('synth') == label:
            return node
#%%
def synthesize_porphyrin(pyrroles, aldehydes):
    for aldehyde, i in zip(aldehydes, range(4)):
        aldehyde.remove_edge('c', 'o')
        aldehyde.remove_node('o')
        aldehyde.nodes['c']['synth'] = f'a{i}c'
        aldehyde.nodes['c']['hcount'] -= 1

    for i in [1, 3]:
        pyrroles[i].nodes['N']['hcount'] = 0

    pyrroles[0].nodes['C1']['synth'] = 'p0c1'
    pyrroles[0].nodes['C4']['synth'] = 'p0c4'

    pyrrole = pyrroles[1]
    pyrrole.edges['C1', 'C2']['order'] = 1
    pyrrole.edges['C3', 'C2']['order'] = 2
    pyrrole.edges['C3', 'C4']['order'] = 1
    pyrrole.edges['N', 'C4']['order'] = 2
    pyrrole.nodes['C1']['synth'] = 'p1c1'
    pyrrole.nodes['C4']['synth'] = 'p1c4'

    pyrrole = pyrroles[3]
    pyrrole.edges['C1', 'C2']['order'] = 1
    pyrrole.edges['C3', 'C2']['order'] = 2
    pyrrole.edges['C3', 'C4']['order'] = 1
    pyrrole.edges['N', 'C1']['order'] = 2
    pyrrole.nodes['C1']['synth'] = 'p3c1'
    pyrrole.nodes['C4']['synth'] = 'p3c4'

    pyrrole = pyrroles[2]
    pyrrole.edges['C1', 'C2']['order'] = 1
    pyrrole.edges['C3', 'C2']['order'] = 2
    pyrrole.edges['C3', 'C4']['order'] = 1
    pyrrole.nodes['C1']['synth'] = 'p2c1'
    pyrrole.nodes['C4']['synth'] = 'p2c4'

    p = disjoint_union(pyrroles[0], aldehydes[0])
    p.add_edge(get_node_by_synth_label(p, 'p0c4'), get_node_by_synth_label(p,
                                                            'a0c'), order=1)
    p = disjoint_union(p, pyrroles[1])
    p.add_edge(get_node_by_synth_label(p, 'a0c'), get_node_by_synth_label(p,
                                                            'p1c1'), order=2)
    p = disjoint_union(p, aldehydes[1])
    p.add_edge(get_node_by_synth_label(p, 'a1c'), get_node_by_synth_label(p,
                                                            'p1c4'), order=1)
    p = disjoint_union(p, pyrroles[2])
    p.add_edge(get_node_by_synth_label(p, 'a1c'), get_node_by_synth_label(p,
                                                            'p2c1'), order=2)
    p = disjoint_union(p, aldehydes[2])
    p.add_edge(get_node_by_synth_label(p, 'a2c'), get_node_by_synth_label(p,
                                                            'p2c4'), order=2)
    p = disjoint_union(p, pyrroles[3])
    p.add_edge(get_node_by_synth_label(p, 'a2c'), get_node_by_synth_label(p,
                                                            'p3c1'), order=1)
    p = disjoint_union(p, aldehydes[3])
    p.add_edge(get_node_by_synth_label(p, 'a3c'), get_node_by_synth_label(p,
                                                            'p3c4'), order=2)
    p.add_edge(get_node_by_synth_label(p, 'a3c'), get_node_by_synth_label(p,
                                                            'p0c1'), order=1)

    for i in range(4):
        for j in [1, 4]:
            label = f'p{i}c{j}'
            node = get_node_by_synth_label(p, label)
            p.nodes[node]['hcount'] = 0

    return p
#%%
p = get_pyrrolle_g()
#%%
p.nodes['C4']
#%%
pyrroles = [get_pyrrolle_g() for i in range(4)]
aldehydes = [get_formaldehyde() for i in range(4)]
test = synthesize_porphyrin(pyrroles, aldehydes)
#%%
plt.clf()
draw(test, with_labels=True)
plt.show()
#%%
pyrrole_g = get_pyrrolle_g()
#%%
pyrrole_g.edges['C1', 'C2']['order']
#%%
adj_view = pyrrole_g.adj
for node in adj_view:
    if node == 'N':
        for key in adj_view['N'].keys():
            print(key)
#%%
test.adj
#%%
bonds_dict = SymDict()
bonds_lines = []
atoms_lines = []

for atom_ix in test.adj:
    element = test.nodes[atom_ix]['element']
    atoms_lines.append(f'0.0 0.0 0.0 {element} 0 0 0 0 0 0 0 0 0 0 0 0')
    atom = test.adj[atom_ix]
    for atom_jx, bond in atom.items():
        if not bonds_dict.get([atom_ix, atom_jx]):
            bonds_dict.set([atom_ix, atom_jx], True)
            bond_line = ''.join(tuple(f'{int_}'.rjust(3, ' ') for int_ in
                                      [atom_ix + 1, atom_jx + 1,
                                       bond['order'], 0]))
            bonds_lines.append(bond_line)
        print(atom_ix, atom_jx, bond['order'])
#%%
for line in bonds_lines:
    print(line)
#%%
m = Chem.MolFromMolFile('porphin.mol', strictParsing=False)

