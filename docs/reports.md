# Reports

`read`, `read_all`, `write` and `convert` return a [`Report`](api.md#medvol.Report) of everything they decided:

- `choices`: what a policy or the data decided (`nifti_form = sform (code 1)`, `geometry = reused from source
  (unchanged)`, ...);
- `warnings`: things that are probably fine but worth knowing (qform and sform disagree, a header of another
  format is not written, ...);
- `dropped`: what the target format cannot store and was left out.

```python
image.meta.user = {"study": "A"}
report = medvol.write(image, "out.nii.gz")
print(report)
```

```text
write nifti out.nii.gz
  choice: geometry = reused from source (unchanged)
  dropped: user: the format has no place for arbitrary metadata
```

With `on_loss="error"`, a drop raises `medvol.LossError` instead, before anything is written:

```python
try:
    medvol.write(image, "out.nii.gz", on_loss="error")
except medvol.LossError as error:
    print(error)            # user: the format has no place for arbitrary metadata
```

Reading also appends the report to the image's history (`action: converted`, the tool and version, the
choices as `settings`, warnings and drops as `warnings`), so a `.mla` file records how it was made:

```python
image.meta.history[-1]
# {'action': 'converted', 'tool': 'medvol 1.0.0', 'from': 'nifti',
#  'settings': {'nifti_form': 'sform (code 1)'}, 'time': '2026-11-06T12:29:50Z'}
```

The [CLI](cli.md) prints the report of every conversion.
