from  torch_geometric_autoscale import *
import pytest
import torch

@pytest.fixture
def dbh():
    # 10 nodes
    # 3  layers
    # 20 hidden
    return DBHistory(10,3,20)

@pytest.fixture
def dbh_nomultiget():
    return DBHistory(10,3,20,multi_get = False)

@pytest.fixture
def v():
    return torch.Tensor(list(range(20)))

@pytest.fixture
def vv():
    return torch.Tensor([list(range(20)),list(range(30,50))])

@pytest.fixture
def vvv():
    return torch.Tensor([[list(range(10,30)),list(range(40,60))],[list(range(20)),list(range(30,50))],[list(range(50,70)),list(range(60,80))]])



def test_key(dbh):
    k = dbh.key(7,2)
    assert k == b'7.2'

def test_db_value_from_tensor_and_back(dbh,v):
    dbv = dbh.db_value_from_tensor(v)
    t = dbh.tensor_from_dbvalue(dbv)
    assert torch.equal(v,t)

def test_reset(dbh,v):
    dbv = dbh.db_value_from_tensor(v)
    dbh.db.put(dbh.key(5,2), dbv)
    t = dbh.tensor_from_dbvalue(dbh.db.get(dbh.key(5,2)))
    assert torch.equal(v,t)
    dbh.reset_parameters()
    t = dbh.tensor_from_dbvalue(dbh.db.get(dbh.key(5,2)))
    v = torch.Tensor([0]*20)
    assert torch.equal(v,t)

def test_push_simple_multiget(dbh,v):
    dbh.push(v,torch.LongTensor([5]), torch.LongTensor([2]))
    v2 = dbh.pull(torch.LongTensor([5]), torch.LongTensor([2]))
    assert torch.equal(v,v2)

def test_push_simple_nomultiget(dbh_nomultiget,v):
    dbh_nomultiget.push(v,torch.LongTensor([5]), torch.LongTensor([2]))
    v2 = dbh_nomultiget.pull(torch.LongTensor([5]), torch.LongTensor([2]))
    assert torch.equal(v,v2)


def test_push_multi_id(dbh,vv):
    dbh.push(vv,n_id = torch.LongTensor([2,5]), layers = torch.LongTensor([2]))
    v2 = dbh.pull(n_id = torch.LongTensor([2,5]), layers = torch.LongTensor([2]))
    assert torch.equal(vv,v2)

def test_push_multi_layer(dbh,vv):
    dbh.push(vv,n_id = torch.LongTensor([2]), layers = torch.LongTensor([0,2]))
    v2 = dbh.pull(n_id = torch.LongTensor([2]), layers = torch.LongTensor([0,2]))
    assert torch.equal(vv,v2)


def test_push_multi_all(dbh,vvv):
    dbh.push(vvv,n_id = torch.LongTensor([2,4,7]), layers = torch.LongTensor([0,2]))
    v2 = dbh.pull(n_id = torch.LongTensor([2,4,7]), layers = torch.LongTensor([0,2]))
    assert torch.equal(vvv,v2)

def test_push_multi_all_list(dbh,vvv):
    dbh.push(vvv,n_id = [2,4,7], layers = [0,2])
    v2 = dbh.pull(n_id = [2,4,7], layers = [0,2])
    assert torch.equal(vvv,v2)

#def test_push_offset_count(dbh):
