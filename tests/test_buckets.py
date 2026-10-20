from hinglish_mt.buckets import group_of


def test_group_of():
    assert group_of("mujhe nahi pata tha") == "Hindi only"
    assert group_of("i am so tired today") == "English only"
    assert group_of("aaj office me bahut kaam tha yaar") == "Mixed, low (CMI up to 20)"
    assert group_of("aaj office me bahut kaam tha, i am so tired") == "Mixed, high (CMI above 20)"
