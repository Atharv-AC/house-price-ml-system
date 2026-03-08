#===========================================
# 🧪 Test 1 — load_model success
#===========================================

def test_load_model_success(monkeypatch):
    """
    Test Case:
    Verify that load_model() correctly loads the latest model file
    and returns the loaded object.

    ? We mock (replace) joblib.load so:
    - No real file is accessed
    - The test runs fast
    - The test is independent of the filesystem
    """


    # *1️⃣ Import the module that contains the function to test
    # We import the loader module from our project.
    # load_model() exists inside this module.
    from house_price_prediction import loader


    # *2️⃣ Create a fake version of joblib.load
    # Instead of actually loading a file from disk,
    # this fake function will be used during the test.
    
    # It receives the file path as argument
    # and returns a fake model object.
    def fake_load(path):

        # ✅ Ensure the correct file is being requested
        # This checks that load_model() is trying to load "latest.joblib"
        assert "latest.joblib" in path

        # ✅ Return a fake model object
        # In real life this would be a trained ML model,
        # but here we just return a simple string.
        return "fake_model"


    # *3️⃣ Replace joblib.load with our fake function                       
    #                                                                     # ! In Production: 
    # monkeypatch temporarily replaces:                                       load_model()
    # loader.joblib.load  --->  fake_load                                         ↓
    #                                                                         joblib.load()
    # So when load_model() calls joblib.load(),                                   ↓
    # it will actually call fake_load() instead.                              Filesystem
    monkeypatch.setattr(loader.joblib, "load", fake_load)



    # *4️⃣ Call the function we want to test
    result = loader.load_model()                                           


    #                                                                          # ! In Test:                                                                                                                  
    # *5️⃣ Verify the result                                                    load_model()                                                                                                              
    #                                                                               ↓                                                                                                  
    # We check whether load_model() returned what fake_load() returned.         fake_load()                                                                                            
    # If yes -> test passes.                                                        ↓                       
    # If not -> test fails.                                                     No filesystem
    assert result == "fake_model"





#===========================================
# 🧪 Test 2 — load_model failure
#===========================================
def test_load_model_failure(monkeypatch):
    from house_price_prediction import loader
    import pytest

    def fake_load(path):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(loader.joblib, "load", fake_load)

    with pytest.raises(FileNotFoundError):
        loader.load_model()        # fake_load("models/latest.joblib")


#================================================
# 🧪 Test 3 — save_model calls joblib.dump
# ==================================================
        """ 
        🎯 Goal:
            Ensure that save_model() correctly calls joblib.dump()
            with the proper model and file path.
        
            ❌ We DO NOT write to the filesystem.
            ✅ We simulate (mock) joblib.dump using monkeypatch.
        """

def test_save_model(monkeypatch):
    from house_price_prediction import loader                                # ! In production: 
    #                                                                            save_model()
    # 📌 Dictionary to capture arguments passed to dump()                            ↓
    # This acts like a spy to record what was called.                            joblib.dump()
    called = {}                                                                  #    ↓
    #                                                                            Filesystem (file created)

    # 🔹 1️⃣ Create Fake dump Function
    # --------------------------------------------------
    # Instead of writing to disk,
    # we store the arguments inside `called`.
    def fake_dump(model, path):
        called["model"] = model
        called["path"] = path



    # 🔹 2️⃣ Replace joblib.dump with fake_dump                                #! In Test:
    # --------------------------------------------------                        save_model()
    # monkeypatch temporarily replaces:                                              ↓
    # loader.joblib.dump  --->  fake_dump                                        fake_dump() 
    #                                                                                ↓
    # So when save_model() calls joblib.dump(),                                  Dictionary capture (no filesystem)
    # it will actually call fake_dump().
    monkeypatch.setattr(loader.joblib, "dump", fake_dump)


    # 🔹 3️⃣ Call the function we want to test
    #? dummy_model and test.joblib are just a string values
    loader.save_model("dummy_model", "test.joblib")



    # 🔹 4️⃣ Verify behavior
    # --------------------------------------------------
    # We check whether save_model() passed
    # the correct arguments to joblib.dump().
    #
    # If yes  → test passes ✅
    # If not → test fails ❌
    assert called["model"] == "dummy_model"
    assert "test.joblib" in called["path"]