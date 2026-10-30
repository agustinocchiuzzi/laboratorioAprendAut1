import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/study_tools.py'
spec = importlib.util.spec_from_file_location('study_tools', PATH)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class StudyPreparationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = study.read_json(study.DEFAULT_PROTOCOL)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def saved_run(self, seed, returns):
        path = self.root / f'seed{seed}'
        path.mkdir()
        config = {**self.protocol['reference'], **self.protocol['budget'],
            'training_seed': seed, 'evaluation_seed': 10000, 'evaluation_episodes': 200,
            'evaluation_exploration': False}
        (path/'config.json').write_text(json.dumps(config))
        (path/'summary.json').write_text(json.dumps(dict(mean_return=sum(returns)/len(returns),
            successes=sum(x >= 200 for x in returns), total_steps=300,
            training_seconds=1.0)))
        (path/'evaluation.csv').write_text('reset_seed,return,success\n'+''.join(
            f'{10000+i},{r},{int(r>=200)}\n' for i,r in enumerate(returns)))
        (path/'training.csv').write_text('episode,length\n'+''.join(f'{i+1},1\n' for i in range(300)))
        (path/'online_weights.pt').touch()
        return path

    def test_manifest_recognizes_integrated_switches(self):
        manifest = study.make_manifest(self.protocol)
        self.assertEqual(manifest['job_count'], 32)
        jobs = {j['job_id']:j for j in manifest['jobs']}
        self.assertEqual(jobs['full_seed42']['missing_agent_switches'], [])
        self.assertEqual(jobs['no_replay_seed42']['missing_agent_switches'], [])
        self.assertEqual(jobs['no_exploration_seed42']['missing_agent_switches'], [])

    def test_incompatible_exploration_and_target_schedules_rejected(self):
        run = self.saved_run(42, [0]*200)
        original = study.read_json(run/'config.json')
        for key, value in [('epsilon_schedule', 'exponential_episodes'), ('target_update', 'soft')]:
            (run/'config.json').write_text(json.dumps({**original, key: value}))
            with self.assertRaisesRegex(ValueError, key + ' differs'):
                study.aggregate(self.protocol, [run])

    def test_overlapping_evaluation_seeds_and_duplicate_training_seeds_rejected(self):
        bad = copy.deepcopy(self.protocol)
        bad['evaluation']['reserved_seed_start'] = 10100
        with self.assertRaises(ValueError): study.validate_protocol(bad)
        bad = copy.deepcopy(self.protocol)
        bad['training_seeds'] = [42,42]
        with self.assertRaises(ValueError): study.validate_protocol(bad)

    def test_aggregation_uses_training_seed_means_and_marks_missing_runs(self):
        first=self.saved_run(42,[0]*200)
        second=self.saved_run(1042,[200]*200)
        report=study.aggregate(self.protocol,[first,second])
        full=report['variants']['full']
        self.assertEqual(full['mean_across_training_seeds'],100)
        self.assertAlmostEqual(full['training_seed_sample_sd'],141.4213562373095)
        self.assertEqual(len(full['per_training_seed']),2)
        self.assertFalse(report['complete'])

    def test_duplicate_policy_or_incompatible_settings_rejected(self):
        run=self.saved_run(42,[0]*200)
        with self.assertRaises(ValueError): study.aggregate(self.protocol,[run,run])
        config=study.read_json(run/'config.json');config['lr']=.0005
        (run/'config.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'lr differs'): study.aggregate(self.protocol,[run])

    def test_wrong_evaluation_seed_and_stale_summary_rejected(self):
        run=self.saved_run(42,[0]*200)
        original=(run/'evaluation.csv').read_text()
        (run/'evaluation.csv').write_text(original.replace('10000,','9999,',1))
        with self.assertRaisesRegex(ValueError,'CSV seeds'): study.aggregate(self.protocol,[run])
        (run/'evaluation.csv').write_text(original)
        summary=study.read_json(run/'summary.json');summary['mean_return']=1
        (run/'summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError,'disagrees'): study.aggregate(self.protocol,[run])

    def test_existing_output_is_never_overwritten(self):
        path=self.root/'existing.json';path.write_text('{}')
        with self.assertRaises(FileExistsError):study.write_new_json(path,{'replacement':True})


if __name__ == '__main__':unittest.main()
