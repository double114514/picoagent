"""Memory trimming tests."""
from picoagent import Memory, Message, Role


def test_add_and_len():
    m = Memory()
    m.add_user("hello")
    assert len(m) == 1
    assert m.messages()[0].role == Role.USER


def test_trim_keeps_system_and_recent():
    m = Memory(max_messages=4)
    m.add(Message(role=Role.SYSTEM, content="sys"))
    for i in range(10):
        m.add_user(f"msg{i}")
    msgs = m.messages()
    assert len(msgs) == 4
    assert msgs[0].role == Role.SYSTEM
    assert msgs[-1].content == "msg9"


def test_clear():
    m = Memory()
    m.add_user("x")
    m.clear()
    assert len(m) == 0
