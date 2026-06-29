from tooltrim.store import FileStore, OutputStore, RedisStore, S3Store, make_ref


def _roundtrip(store):
    ref = store.put("hello world, the secret is 42")
    assert store.get(ref) == "hello world, the secret is 42"
    # content-addressed: same content -> same ref
    assert store.put("hello world, the secret is 42") == ref
    # expand slicing (shared BaseStore logic)
    assert store.expand(ref, start=0, length=5) == "hello"
    assert store.expand(ref, start=6) == "world, the secret is 42"
    assert store.get("deadbeef") is None
    assert store.expand("deadbeef") is None


def test_make_ref_stable_and_short():
    assert make_ref("abc") == make_ref("abc")
    assert len(make_ref("abc")) == 8
    assert make_ref("abc") != make_ref("abd")


def test_inmemory_store_roundtrip():
    _roundtrip(OutputStore())


def test_inmemory_store_lru_eviction():
    s = OutputStore(max_entries=2)
    a = s.put("aaaa")
    b = s.put("bbbb")
    s.get(a)            # touch a so b is now least-recently-used
    s.put("cccc")       # evicts b
    assert s.get(a) is not None
    assert s.get(b) is None


def test_file_store_roundtrip(tmp_path):
    _roundtrip(FileStore(str(tmp_path / "store")))


def test_file_store_persists_across_instances(tmp_path):
    root = str(tmp_path / "shared")
    ref = FileStore(root).put("durable content here")
    # a *second* process/worker pointed at the same dir can expand it
    assert FileStore(root).get(ref) == "durable content here"


class _FakeRedis:
    def __init__(self):
        self.kv = {}

    def set(self, key, value, ex=None):
        self.kv[key] = value

    def get(self, key):
        return self.kv.get(key)


def test_redis_store_roundtrip_with_injected_client():
    _roundtrip(RedisStore(client=_FakeRedis()))


def test_redis_store_uses_prefix():
    fake = _FakeRedis()
    s = RedisStore(client=fake, prefix="tt:")
    ref = s.put("xyz")
    assert ("tt:" + ref) in fake.kv


class _FakeBody:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


class _FakeS3:
    def __init__(self):
        self.objs = {}

    def put_object(self, Bucket, Key, Body):  # noqa: N803 (boto3 signature)
        self.objs[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):  # noqa: N803
        if (Bucket, Key) not in self.objs:
            raise KeyError("no such key")
        return {"Body": _FakeBody(self.objs[(Bucket, Key)])}


def test_s3_store_roundtrip_with_injected_client():
    _roundtrip(S3Store(bucket="b", client=_FakeS3()))


def test_compressor_accepts_pluggable_store(tmp_path):
    from tooltrim import ToolCompressor

    tc = ToolCompressor(max_tokens=80, store=FileStore(str(tmp_path / "s")))
    res = tc.compress("noise " * 2000 + "the code is ALPHA9", query="code")
    assert res.ref is not None
    assert "ALPHA9" in tc.expand(res.ref)
