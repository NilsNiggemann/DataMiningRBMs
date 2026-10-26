# Running Claude Code in tmux (survives closing SSH)

Without tmux, closing the SSH session kills `claude` (SIGHUP), and Remote Control and
probably its background jobs die with it.

```bash
tmux new -s claude                      # 1. new named session (after ssh-ing in)
cd ~/Projects/DataMiningRBMs && claude  # 2. start Claude inside it
                                        #    (or `claude --resume` / `claude --continue`)
# 3. detach: Ctrl-b then d   -> Claude keeps running, SSH can be closed
tmux attach -t claude                   # 4. reattach after the next ssh login
tmux ls                                 #    list sessions
```

- Scroll back: `Ctrl-b` `[`, then arrows/PgUp; `q` to leave.
- End: `/exit` in Claude, then `exit` in the shell.
