"""NotificationStore keeps a bounded, newest-first list.

The bell badge is the unread count. Growing without a cap, or appending
instead of inserting at the front, would either leak every message for the
session or show the oldest error as the latest one.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.notification_panel import NotificationStore  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def test_store_keeps_the_newest_fifty(qapp):
    store = NotificationStore()
    added = []
    store.notification_added.connect(lambda: added.append(1))

    for i in range(55):
        store.add(f"m{i}", "error" if i % 2 else "info")

    entries = store.entries()
    assert len(added) == 55
    assert len(entries) == NotificationStore.MAX_ENTRIES == 50
    assert entries[0].message == "m54"
    assert entries[-1].message == "m5"
    assert [entry.category for entry in entries[:2]] == ["info", "error"]
    assert store.unread_count() == 50

    store.mark_all_read()
    assert store.unread_count() == 0
    assert len(store.entries()) == 50

    store.clear()
    assert store.entries() == []
    assert store.unread_count() == 0
    assert len(added) == 56
