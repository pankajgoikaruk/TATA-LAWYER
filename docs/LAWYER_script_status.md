# LAWYER script status

The uploaded `scripts/lawyer_refine_weak_labels_v08.py` is already config-driven.

It expects:

```text
--config configs/lawyer_v08_human_talk.json
```

So it does not need another rewrite unless you want to change the JSON schema.

For reuse on a new dataset, create a different config file and pass it through `--config`.
