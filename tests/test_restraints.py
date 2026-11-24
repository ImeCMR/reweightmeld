import mdtraj
import numpy as np

from reweightmeld.restraints import RestraintScorer, load_restraints, parse_spec, read_distance_file


def test_parse_groups_kinds_and_defaults(tmp_path):
    (tmp_path / "NOE.dat").write_text("1 CA 4 CA 5.0\n1 CA 5 CA 6.0\n\n2 O 5 N 3.0\n\n3 CA 6 CA 4.5\n")
    (tmp_path / "hyd.dat").write_text("1 CA 6 CA\n")
    g = read_distance_file(tmp_path / "NOE.dat", "noe")
    assert [len(x.lines) for x in g] == [2, 1, 1]
    assert [x.kind for x in g] == ["noe", "hbond", "noe"]
    assert g[0].lines[0] == (0, "CA", 3, "CA", 5.0)
    assert parse_spec("hyd.dat:contact:5.5") == ("hyd.dat", "contact", 5.5)
    rs = load_restraints(["NOE.dat:noe", "hyd.dat:contact:5.5"], tmp_path)
    assert len(rs) == 4 and rs[-1].kind == "contact" and rs[-1].lines[0][4] == 5.5


def test_group_satisfied_by_any_line(tmp_path):
    (tmp_path / "r.dat").write_text("1 CA 2 CA 5.0\n1 CA 3 CA 5.0\n\n1 CA 3 CA 5.0\n\n1 CA 9 CA 5.0\n")
    top = mdtraj.Topology()
    ch = top.add_chain()
    for i in range(3):
        top.add_atom("CA", mdtraj.element.carbon, top.add_residue("ALA", ch))
    xyz = np.array([[[0, 0, 0], [0.4, 0, 0], [0.8, 0, 0]]], dtype=np.float32)   # nm: 4 and 8 A
    sc = RestraintScorer(read_distance_file(tmp_path / "r.dat"), top)
    assert sc.n_missing == 1                                     # residue 9 does not exist
    assert sc.satisfied(mdtraj.Trajectory(xyz, top)).tolist() == [[True, False, False]]
    assert sc.separation.tolist() == [1, 2, 8]
