cd /home/jiankais/lustre_jiankais/Programs/hojune/Psi0-hojune/.runs/finetune/ || exit 1

for d in \
  psi0-18act-eef-full.real.flow1000.cosine.lr5.0e-05.b512.gpus8.2604192206 \
  psi0-18act-joint-full.real.flow1000.cosine.lr5.0e-05.b512.gpus8.2604192219 \
  psi0-36act-joint-aligned-full.real.flow1000.cosine.lr5.0e-05.b512.gpus8.2604192219
do
  case "$d" in
    psi0-18act-eef-full.real.flow1000.cosine.lr5.0e-05.b512.gpus8.2604192206)
      out="stanford:Hojune/ckpts/psi0-18act-eef-full-40000"
      ;;
    psi0-18act-joint-full.real.flow1000.cosine.lr5.0e-05.b512.gpus8.2604192219)
      out="stanford:Hojune/ckpts/psi0-18act-joint-full-40000"
      ;;
    psi0-36act-joint-aligned-full.real.flow1000.cosine.lr5.0e-05.b512.gpus8.2604192219)
      out="stanford:Hojune/ckpts/psi0-36act-joint-aligned-full-40000"
      ;;
    *)
      echo "Unknown run directory: $d"
      continue
      ;;
  esac

  echo "Uploading $d -> $out"

  rclone copy "$d/checkpoints/ckpt_40000" "$out" \
    -P # --drive-chunk-size 64M --multi-thread-streams 4 --bwlimit 3M --transfers 4 --checkers 8 --tpslimit 1 --tpslimit-burst 1

  rclone copy "$d/argv.txt" "$out" \
    -P #--tpslimit 1 --tpslimit-burst 1

  rclone copy "$d/envs.txt" "$out" \
    -P #--tpslimit 1 --tpslimit-burst 1

  rclone copy "$d/run_config.json" "$out" \
    -P #--tpslimit 1 --tpslimit-burst 1

  sleep 20
done