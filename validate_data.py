"""Independent flat PCFG SET executor and dataset integrity checks.

Setup validation only: no neural model receives executor outputs.
"""
import json
from itertools import zip_longest
from pathlib import Path
import unittest

UNARY = {
    'copy': lambda x: list(x),
    'reverse': lambda x: x[::-1],
    'shift': lambda x: x[1:] + x[:1],
    'echo': lambda x: x + x[-1:],
    'swap_first_last': lambda x: x[-1:] + x[1:-1] + x[:1],
    'repeat': lambda x: x * 2,
}
BINARY = {
    'append': lambda x, y: x + y,
    'prepend': lambda x, y: y + x,
    'remove_first': lambda x, y: list(y),
    'remove_second': lambda x, y: list(x),
}


def untwin(token):
    return token[:-5] if token.endswith('_twin') else token


def execute(text):
    tokens = text.split()

    def expression(i):
        if i >= len(tokens) or tokens[i] == ',':
            raise ValueError('Missing argument')
        op = untwin(tokens[i])
        if op in UNARY:
            arg, j = expression(i + 1)
            return UNARY[op](arg), j
        if op in BINARY:
            left, j = expression(i + 1)
            if j >= len(tokens) or tokens[j] != ',':
                raise ValueError('Missing binary separator')
            right, k = expression(j + 1)
            return BINARY[op](left, right), k
        j = i
        while j < len(tokens) and tokens[j] != ',':
            if untwin(tokens[j]) in UNARY.keys() | BINARY.keys():
                raise ValueError('Function embedded in string literal')
            j += 1
        return tokens[i:j], j

    result, end = expression(0)
    if end != len(tokens):
        raise ValueError('Unconsumed input')
    return ' '.join(result)


class ExecutorTests(unittest.TestCase):
    def test_primitives(self):
        cases = {
            'copy A B C': 'A B C', 'reverse A B C': 'C B A',
            'shift A B C': 'B C A', 'echo A B C': 'A B C C',
            'swap_first_last A B C': 'C B A', 'repeat A B C': 'A B C A B C',
            'append A B , C D': 'A B C D', 'prepend A B , C D': 'C D A B',
            'remove_first A B , C D': 'C D', 'remove_second A B , C D': 'A B',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(execute(text), expected)

    def test_nested(self):
        self.assertEqual(execute('reverse append echo A B , shift C D'), 'C D B B A')
        self.assertEqual(execute('append prepend A B , C D , repeat E F'), 'C D A B E F E F')
        self.assertEqual(execute('remove_second reverse A B , append C D , E F'), 'B A')

    def test_synonyms(self):
        for op in ['swap_first_last', 'repeat', 'append', 'remove_second']:
            text = op + (' A B , C D' if op in BINARY else ' A B')
            self.assertEqual(execute(text), execute(text.replace(op, op + '_twin', 1)))

    def test_malformed(self):
        for text in ['', 'append A B', 'reverse', 'copy A B , C D', 'append , C D']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                execute(text)


def validate(root):
    results = {}
    for source in sorted(root.rglob('*.src')):
        # Exception targets intentionally differ from standard semantics.
        if 'overgeneralisation' in source.parts and '_original' not in source.stem:
            continue
        target = source.with_suffix('.tgt')
        if not target.exists():
            raise AssertionError('Missing target: ' + str(target))
        total, mismatches, examples = 0, 0, []
        with source.open() as src, target.open() as tgt:
            for n, (s, t) in enumerate(zip_longest(src, tgt), 1):
                if s is None or t is None:
                    raise AssertionError('Unequal line counts: ' + str(source))
                total += 1
                if execute(s) != t.strip():
                    mismatches += 1
                    if len(examples) < 3:
                        examples.append({'line': n, 'source': s.strip(), 'expected': t.strip(), 'actual': execute(s)})
        results[str(source.relative_to(root))] = {'examples': total, 'mismatches': mismatches, 'first_mismatches': examples}
    return results


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExecutorTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    report = validate(Path(__file__).resolve().parent / 'original/data/pcfgset')
    output = Path(__file__).resolve().parent / 'validation/hupkes/data-validation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('Validated', sum(v['examples'] for v in report.values()), 'pairs across', len(report), 'files')
    mismatch_count = sum(v['mismatches'] for v in report.values())
    print('Mismatches:', mismatch_count)
    raise SystemExit(bool(mismatch_count))
