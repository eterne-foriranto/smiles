#%%
from math import acos, cos, pi, sin
from matplotlib import pyplot as plt
from networkx import Graph, compose, disjoint_union, draw, union
from pysmiles import write_smiles
from rdkit.Avalon.pyAvalonTools import GetCanonSmiles
from rdkit import Chem
#%%
standard_length = 1
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
def cartesian2polar(x, y):
    r = (x * x + y * y) ** 0.5
    if not r:
        return 0, 0
    arc_cosine = acos(x / r)
    if y:
        phi = arc_cosine
    else:
        phi = 2 * pi - arc_cosine
    return r, phi


def polar2cartesian(r, phi):
    x = r * cos(phi)
    y = r * sin(phi)
    return x, y


def get_coord_diff(from_, to):
    delta_x = to[0] - from_[0]
    delta_y = to[1] - from_[1]
    return delta_x, delta_y


def apply_diff(coords, diff):
    new_x = coords[0] + diff[0]
    new_y = coords[1] + diff[1]
    return new_x, new_y


def compare(val_1, ratio, val_2):
    if ratio == '<':
        return val_1 < val_2
    else:
        return val_1 > val_2


class Molecule:
    def __init__(self, init_dict={'atoms': [], 'bonds': SymDict()}):
        self.data = init_dict

    def get_center(self):
        if hasattr(self, 'center'):
            return self.center
        else:
            mean_x, mean_y = 0, 0
            atoms = self.data['atoms']
            for atom in atoms:
                mean_x += atom['coords'][0]
                mean_y += atom['coords'][1]

            mean_x /= len(atoms)
            mean_y /= len(atoms)
            self.center = mean_x, mean_y

            return self.center

    def rotate(self, angle):
        atoms = self.data['atoms']
        for atom in atoms:
            rel_cartesian = get_coord_diff(atom['coords'], self.get_center())
            polar = cartesian2polar(*rel_cartesian)
            new_polar = polar[0], polar[1] + angle
            rel_cartesian = polar2cartesian(*new_polar)
            atom['coords'] = apply_diff(self.get_center(), rel_cartesian)

    def get_most_factory(coord_ix, ratio):

        def fun2ret(self):
            atoms = self.data['atoms']
            extreme_atom = atoms[0]
            extreme_coord = atoms[0]['coords'][coord_ix]
            for atom in atoms:
                if compare(atom['coords'][coord_ix], ratio, extreme_coord):
                    extreme_atom = atom
                    extreme_coord = atom['coords'][coord_ix]
            return extreme_atom

        return fun2ret

    get_most_right_atom = get_most_factory(0, '>')
    get_most_left_atom = get_most_factory(0, '<')
    get_most_upper_atom = get_most_factory(1, '>')
    get_most_down_atom = get_most_factory(1, '<')

    def move(self, vector):
        atoms = self.data['atoms']
        for atom in atoms:
            atom['coords'] = apply_diff(atom['coords'], vector)

        if hasattr(self, 'center'):
            self.center = apply_diff(self.center, vector)

    def place2side_factory(extreme_fun_1, extreme_fun_2, diff):
        def fun2ret(self, other_molecule):
            other_molecule_extreme_coords = getattr(other_molecule,
                extreme_fun_1)()['coords']
            this_molecule_target_extreme_coords = apply_diff(
                other_molecule_extreme_coords, diff)
            this_molecule_actual_extreme_coords = getattr(self,
                                                          extreme_fun_2)()[
                'coords']
            vector4moving = get_coord_diff(this_molecule_actual_extreme_coords,
                                        this_molecule_target_extreme_coords)
            self.move(vector4moving)

        return fun2ret

    place2right = place2side_factory('get_most_right_atom',
                                     'get_most_left_atom', (standard_length, 0))
    place2left = place2side_factory('get_most_left_atom',
                                     'get_most_right_atom', (-standard_length,
                                                             0))
    place2down = place2side_factory('get_most_down_atom',
                                    'get_most_upper_atom', (0,
                                                            -standard_length))
    place2up = place2side_factory('get_most_upper_atom',
                                  'get_most_down_atom', (0, standard_length))

    def get_atom_ix_by_label(self, label):
        atoms = self.data['atoms']
        for atom, ix in zip(atoms, range(len(atoms))):
            if atom.get('label') == label:
                return ix

    def dump_mol(self, file, title):
        atoms = self.data['atoms']
        bonds_inner_dict = self.data['bonds'].inner_dict
        atoms_num_str = str(len(atoms)).rjust(3, ' ')
        bonds_num_str = str(len(bonds_inner_dict)).rjust(3, ' ')
        counts_line = atoms_num_str + bonds_num_str +\
                      '  0  0  0  0  0  0  0  0999 V2000'
        atoms_lines = []
        for atom in atoms:
            x_str = str(round(atom['coords'][0], 4)).rjust(10, ' ')
            y_str = str(round(atom['coords'][1], 4)).rjust(10, ' ')
            z_str = '    0.0000'
            element_str = atom['element'].rjust(3, ' ')
            mass_diff_str = ' 0'
            charge_str = str(atom.get('charge', 0)).rjust(3, ' ')
            trailing_str = '  0  0  0  0  0  0  0  0  0  0'
            atom_line = x_str + y_str + z_str + ' ' + element_str + \
                        mass_diff_str + charge_str + trailing_str
            atoms_lines.append(atom_line)
        bonds_lines = []
        for key, value in bonds_inner_dict.items():
            bond_str = ''.join(tuple(str(atom_ix + 1).rjust(3, ' ') for
                                     atom_ix in tuple(sorted(key))))
            order_str = str(value).rjust(3, ' ')
            trailing_str = '  0'
            bond_line = bond_str + order_str + trailing_str
            bonds_lines.append(bond_line)
        lines = [title, ' ', ' ', counts_line, *atoms_lines, *bonds_lines,
                 'M  END']
        file.write('\n'.join(lines))


def unite_molecules(molecules):
    res_molecule = Molecule()
    for molecule in molecules:
        atoms_num = len(res_molecule.data['atoms'])
        res_molecule.data['atoms'].extend(molecule.data['atoms'])
        bonds = molecule.data.get('bonds')
        if bonds:
            for bond, order in bonds.inner_dict.items():
                res_molecule.data['bonds'].set([atom_ix + atoms_num for atom_ix
                                            in bond], order)
    return res_molecule


def get_n_cycle(n, start_angle):
    iterator_ = range(n)
    bonds = SymDict()
    for i in iterator_:
        bonds.set([i % n, (i + 1) % n], 1)
    return Molecule({'atoms': [{'coords': polar2cartesian(standard_length,
                                                          start_angle + 2 *
                                                          pi * i / n),
                                'element': 'C'} for i in iterator_], 'bonds':
        bonds})


def get_pyrrole():
    pyrrole = get_n_cycle(5, -pi / 4)
    pyrrole.data['atoms'][0]['element'] = 'N'
    pyrrole.data['bonds'].set([1, 2], 2)
    pyrrole.data['bonds'].set([3, 4], 2)
    return pyrrole



#%%
pyrrole = get_pyrrole()
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
    for aldehyde, pyrrole, i in zip(aldehydes, pyrroles, range(4)):
        aldehyde.data['atoms'][aldehyde.get_atom_ix_by_label('mezo')][
            'label'] = f'a{i}'
        pyrrole.data['atoms'][1]['label'] = f'p{i}c1'
        pyrrole.data['atoms'][4]['label'] = f'p{i}c4'
        aldehyde.rotate(-pi * i / 2)
        pyrrole.rotate(-pi * i / 2)

    pyrroles[1].data['bonds'].set([1, 2], 1)
    pyrroles[1].data['bonds'].set([3, 2], 2)
    pyrroles[1].data['bonds'].set([3, 4], 1)
    pyrroles[1].data['bonds'].set([0, 4], 2)

    pyrroles[3].data['bonds'].set([1, 2], 1)
    pyrroles[3].data['bonds'].set([3, 2], 2)
    pyrroles[3].data['bonds'].set([3, 4], 1)
    pyrroles[3].data['bonds'].set([0, 1], 2)

    pyrroles[2].data['bonds'].set([1, 2], 1)
    pyrroles[2].data['bonds'].set([3, 2], 2)
    pyrroles[2].data['bonds'].set([3, 4], 1)

    aldehydes[0].place2right(pyrroles[0])
    pyrroles[1].place2right(aldehydes[0])
    aldehydes[1].place2down(pyrroles[1])
    pyrroles[2].place2down(aldehydes[1])
    aldehydes[2].place2left(pyrroles[2])
    pyrroles[3].place2left(aldehydes[2])
    aldehydes[3].place2up(pyrroles[3])

    porphyrin = unite_molecules([*pyrroles, *aldehydes])

    porphyrin.data['bonds'].set([porphyrin.get_atom_ix_by_label('p0c4'),
                                 porphyrin.get_atom_ix_by_label('a0')], 1)
    porphyrin.data['bonds'].set([porphyrin.get_atom_ix_by_label('p1c1'),
                                 porphyrin.get_atom_ix_by_label('a0')], 2)
    porphyrin.data['bonds'].set([porphyrin.get_atom_ix_by_label('p1c4'),
                                 porphyrin.get_atom_ix_by_label('a1')], 1)
    porphyrin.data['bonds'].set([porphyrin.get_atom_ix_by_label('p2c1'),
                                 porphyrin.get_atom_ix_by_label('a1')], 2)
    porphyrin.data['bonds'].set([porphyrin.get_atom_ix_by_label('p2c4'),
                                 porphyrin.get_atom_ix_by_label('a2')], 2)
    porphyrin.data['bonds'].set([porphyrin.get_atom_ix_by_label('p3c1'),
                                 porphyrin.get_atom_ix_by_label('a2')], 1)
    porphyrin.data['bonds'].set([porphyrin.get_atom_ix_by_label('p3c4'),
                                 porphyrin.get_atom_ix_by_label('a3')], 2)
    porphyrin.data['bonds'].set([porphyrin.get_atom_ix_by_label('p0c1'),
                                 porphyrin.get_atom_ix_by_label('a3')], 1)

    return porphyrin
#%%
pyrroles = [get_pyrrole() for i in range(4)]
aldehydes = [Molecule({'atoms': [{'coords': (0, 0), 'element': 'C', 'label':
    'mezo'}]}) for i in range(4)]
porphin = synthesize_porphyrin(pyrroles, aldehydes)
#%%
with open('try_porphin.mol', 'w') as f:
    # noinspection SpellCheckingInspection
    porphin.dump_mol(f, 'try porphin')
#%%
plt.clf()
plt.plot([atom['coords'] for atom in pyrrole.data['atoms']])
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
m = Chem.MolFromMolFile('try_porphin.mol', strictParsing=False)
#%%
smi = write_smiles(test)
#%%
smi = Chem.MolToSmiles(m)
#%%
print(smi)
