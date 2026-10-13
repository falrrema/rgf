"""Regression tests for checkpoint choice and pickle relocation.

PYTHONPATH=python-package RGF_BINARY=/path/to/rgf python3.12 -m unittest discover -s python-package/tests -p test_persistence_parallel.py -v
"""
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from sklearn.base import clone

from rgf import utils
from rgf.rgf_model import RGFExecuter, RGFRegressor, RGFClassifier


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='rgf-persistence-')
        cls.root = Path(cls.directory.name)
        cls.binary = os.environ.get('RGF_BINARY')
        if cls.binary is None:
            config = utils.Config()
            if not config.RGF_AVAILABLE:
                cls.directory.cleanup()
                raise unittest.SkipTest('RGF executable unavailable')
            cls.binary = config.RGF_PATH
        utils.Config.TEMP_PATH = str(cls.root)
        utils.Config.RGF_PATH = cls.binary
        utils.Config.RGF_AVAILABLE = True
        utils.Config.FASTRGF_AVAILABLE = False
        utils.CONFIG = utils.Config()
        rng = np.random.default_rng(17)
        cls.values = rng.integers(0, 5, (512, 16)).astype(float)
        cls.target = .2*cls.values[:, 0] - .1*cls.values[:, 2]
        cls.model = RGFRegressor(algorithm='RGF_Sib', max_leaf=20, test_interval=5, split_threads=2).fit(cls.values, cls.target)
        cls.expected = cls.model.predict(cls.values)
        cls.payload = pickle.dumps(cls.model)

    @classmethod
    def tearDownClass(cls):
        utils.cleanup()
        utils.UUIDS.clear()
        cls.directory.cleanup()

    def test_numeric_checkpoint_and_unrelated_files(self):
        executor = RGFExecuter(config=utils.CONFIG)
        executor._set_paths()
        for suffix in ['-09','-99','-100','-100.backup','.log','-latest']:
            Path(executor._model_file_loc+suffix).write_bytes(b'fixture')
        Path(executor._model_file_loc+'-999').mkdir()
        executor._find_model_file()
        self.assertEqual(executor._model_file,executor._model_file_loc+'-100')
        Path(executor._model_file_loc+'-999').rmdir()

    def test_missing_checkpoint(self):
        executor=RGFExecuter(config=utils.CONFIG)
        executor._set_paths()
        with self.assertRaisesRegex(Exception,'not found'):
            executor._find_model_file()

    def test_independent_loads_and_cleanup(self):
        first,second=pickle.loads(self.payload),pickle.loads(self.payload)
        self.assertNotEqual(first.estimators_[0]._model_file,second.estimators_[0]._model_file)
        np.testing.assert_array_equal(first.predict(self.values),self.expected)
        first.cleanup()
        np.testing.assert_array_equal(second.predict(self.values),self.expected)
        np.testing.assert_array_equal(self.model.predict(self.values),self.expected)
        second.cleanup()

    def test_cleanup_uses_own_directory(self):
        model=pickle.loads(self.payload)
        path=Path(model.estimators_[0]._model_file)
        previous=utils.Config.TEMP_PATH
        alternate=self.root/'alternate'
        alternate.mkdir()
        try:
            utils.Config.TEMP_PATH=str(alternate)
            model.cleanup()
            self.assertFalse(path.exists())
        finally:
            utils.Config.TEMP_PATH=previous

    def test_relocation_fresh_process_and_refit(self):
        estimator=self.model.estimators_[0]
        state=estimator.__getstate__()
        # Emulate an old pickle from a machine whose model directory no longer exists.
        state['_model_file']='/nonexistent/old-machine/model-99'
        state['_file_prefix']='old-machine-prefix'
        state['config']=utils.Config()
        state['config'].RGF_PATH='/nonexistent/old-machine/rgf'
        state['config'].TEMP_PATH='/nonexistent/old-machine/tmp'
        # Store state directly: the original __getstate__ cannot reopen its missing file.
        state_path=self.root/'old-state.pkl';state_path.write_bytes(pickle.dumps(state))
        model_path=self.root/'estimator.pkl';model_path.write_bytes(self.payload)
        np.save(self.root/'query.npy',self.values)
        child=self.root/'destination';child.mkdir()
        code='''
import os,pickle,sys
from pathlib import Path
import numpy as np
from rgf import utils
from rgf.rgf_model import RGFExecuter
root=Path(sys.argv[1]); destination=root/'destination'
utils.Config.TEMP_PATH=str(destination)
utils.Config.RGF_PATH=sys.argv[2]
utils.Config.RGF_AVAILABLE=True;utils.Config.FASTRGF_AVAILABLE=False
state=pickle.loads((root/'old-state.pkl').read_bytes())
restored=object.__new__(RGFExecuter);restored.__setstate__(state)
assert Path(restored._model_file).parent==destination
assert restored.config.RGF_PATH==sys.argv[2]
values=np.load(root/'query.npy')
np.save(root/'restored.npy',restored.predict(values))
model=pickle.loads((root/'estimator.pkl').read_bytes())
assert utils.CONFIG.TEMP_PATH==str(destination)
model.fit(values,.2*values[:,0]-.1*values[:,2])
assert Path(model.estimators_[0]._model_file).parent==destination
'''
        subprocess.run([sys.executable,'-c',code,str(self.root),self.binary],check=True,capture_output=True,text=True)
        np.testing.assert_array_equal(np.load(self.root/'restored.npy'),self.expected)
        self.assertFalse(Path('/nonexistent/old-machine').exists())

    def test_save_model_returns_self_and_native_bytes(self):
        destination=self.root/'saved.rgf'
        self.assertIs(self.model.save_model(str(destination)),self.model)
        self.assertEqual(destination.read_bytes(),Path(self.model.estimators_[0]._model_file).read_bytes())

    def test_original_version_pickle(self):
        original = os.environ.get('RGF_ORIGINAL_PACKAGE')
        if not original:
            self.skipTest('Set RGF_ORIGINAL_PACKAGE to the pinned original python-package')
        source = self.root / 'original-version'
        source.mkdir()
        code = """
import pickle,sys
from pathlib import Path
import numpy as np
from rgf import utils
from rgf.rgf_model import RGFRegressor
root=Path(sys.argv[1])
utils.Config.TEMP_PATH=str(root)
utils.Config.RGF_PATH=sys.argv[2]
utils.Config.RGF_AVAILABLE=True;utils.Config.FASTRGF_AVAILABLE=False
values=np.arange(1024).reshape(128,8)%5
model=RGFRegressor(algorithm='RGF_Sib',max_leaf=10).fit(values,values[:,0])
np.save(root/'expected.npy',model.predict(values))
(root/'old.pkl').write_bytes(pickle.dumps(model))
"""
        environment=dict(os.environ,PYTHONPATH=original)
        subprocess.run([sys.executable,'-c',code,str(source),self.binary],env=environment,
                       check=True,capture_output=True,text=True)
        restored=pickle.loads((source/'old.pkl').read_bytes())
        self.assertEqual(restored.split_threads,1)
        np.testing.assert_array_equal(restored.predict(np.arange(1024).reshape(128,8)%5),
                                      np.load(source/'expected.npy'))
        self.assertEqual(Path(restored.estimators_[0]._model_file).parent,self.root)
        restored.cleanup()

    def test_actual_hundredth_checkpoint(self):
        model=RGFRegressor(algorithm='RGF_Sib',max_leaf=256,test_interval=1,
                           min_samples_leaf=1,split_threads=2).fit(self.values,self.target)
        executor=model.estimators_[0]
        files=list(self.root.glob(executor._file_prefix+'.model-[0-9]*'))
        self.assertGreaterEqual(len(files),100)
        expected=max(files,key=lambda path:int(path.name.rsplit('-',1)[1]))
        self.assertEqual(Path(executor._model_file),expected)
        model.cleanup()

    def test_unfitted_pickle_and_legacy_parameters(self):
        model=RGFRegressor()
        restored=pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.split_threads,1)
        state=model.__getstate__();state.pop('split_threads')
        restored.__setstate__(state)
        self.assertEqual(restored.get_params()['split_threads'],1)

    def test_threads_validation_and_clone(self):
        for value in [0,-1,1.5,True]:
            with self.assertRaisesRegex(ValueError,'split_threads'):
                RGFRegressor(split_threads=value).fit(self.values,self.target)
        self.assertEqual(clone(self.model).split_threads,2)

    def test_binary_classifier_pickle(self):
        model=RGFClassifier(algorithm='RGF_Sib',max_leaf=20,split_threads=2,n_jobs=1)
        model.fit(self.values,self.target>np.median(self.target))
        restored=pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(model.predict_proba(self.values),restored.predict_proba(self.values))
        self.assertIs(model.save_model(str(self.root/'classifier.rgf')),model)
        model.cleanup();restored.cleanup()

    def test_multiclass_training_workers(self):
        model=RGFClassifier(algorithm='RGF_Sib',max_leaf=20,split_threads=2,n_jobs=2)
        model.fit(self.values,np.arange(len(self.values))%3)
        restored=pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(model.predict_proba(self.values),restored.predict_proba(self.values))
        model.cleanup();restored.cleanup()
