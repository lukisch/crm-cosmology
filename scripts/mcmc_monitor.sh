#!/bin/bash
# MCMC Monitor: checks every 30 min, sends Telegram + pushes to GitHub when done

BOT_TOKEN="***TELEGRAM_TOKEN_REMOVED***"
CHAT_ID="595767047"
LOG="/home/cfm-cosmology/results/mcmc_resume_log.txt"
GIT_REPO="/home/cfm-cosmology-git"

send_telegram() {
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d chat_id="${CHAT_ID}" \
        -d text="${msg}" \
        -d parse_mode="Markdown" > /dev/null 2>&1
}

push_results() {
    echo "Pushing results to GitHub..."
    cd "$GIT_REPO"
    
    # Copy all result files
    cp /home/cfm-cosmology/results/cfm_fR_mcmc_chain.npz results/ 2>/dev/null
    cp /home/cfm-cosmology/results/mcmc_resume_log.txt results/ 2>/dev/null
    cp /home/cfm-cosmology/results/cfm_fR_mcmc_chain_checkpoint_*.npz results/ 2>/dev/null
    cp /tmp/cfm_fR_mcmc_summary.txt results/ 2>/dev/null
    
    # Update status
    echo "MCMC COMPLETED: $(date -u)" > results/mcmc_status.txt
    echo "Server: CCX33 (8 cores), 48 walkers, 5000 steps" >> results/mcmc_status.txt
    tail -30 /home/cfm-cosmology/results/mcmc_resume_log.txt >> results/mcmc_status.txt
    
    git add results/
    git commit -m "MCMC results: 48 walkers, 5000 steps, extended run complete"
    git push origin main
    echo "Push complete."
}

push_checkpoint() {
    cd "$GIT_REPO"
    cp /home/cfm-cosmology/results/mcmc_resume_log.txt results/ 2>/dev/null
    cp /home/cfm-cosmology/results/cfm_fR_mcmc_chain_checkpoint_*.npz results/ 2>/dev/null
    
    echo "Status update: $(date -u)" > results/mcmc_status.txt
    tail -5 "$LOG" >> results/mcmc_status.txt
    
    git add results/ 2>/dev/null
    git diff --cached --quiet || {
        git commit -m "MCMC checkpoint: $(tail -1 "$LOG" | head -c 80)"
        git push origin main
    }
}

# Initial notification
send_telegram "🖥️ *MCMC Monitor gestartet*
Server: $(hostname) ($(nproc) Kerne)
GitHub-Push: aktiviert
Pruefe alle 30 Min, Checkpoint-Push alle 3h."

CYCLES=0

while true; do
    sleep 1800  # 30 Minuten
    CYCLES=$((CYCLES + 1))

    if [ ! -f "$LOG" ]; then
        continue
    fi

    LAST=$(tail -1 "$LOG")
    
    # Check if finished
    if echo "$LAST" | grep -qE "MCMC COMPLETE|Production complete|Total time|saved to"; then
        send_telegram "✅ *MCMC FERTIG!*

$(tail -20 "$LOG")

Pushe Ergebnisse zu GitHub..."
        
        push_results
        
        send_telegram "📦 *Ergebnisse auf GitHub gepusht!*
https://github.com/lukisch/cfm-cosmology
Repo: results/

⚠️ SERVER LOESCHEN nicht vergessen!"
        
        if [ -f /tmp/cfm_fR_mcmc_summary.txt ]; then
            send_telegram "📊 *Summary:*
$(head -30 /tmp/cfm_fR_mcmc_summary.txt)"
        fi
        break
    fi

    # Check if crashed
    if ! ps aux | grep -v grep | grep -q "run_full_mcmc"; then
        send_telegram "❌ *MCMC ABGESTUERZT!*
$(tail -5 "$LOG")

SSH: ssh -i ~/.ssh/id_ed25519_mcmc root@46.225.30.98"
        push_checkpoint
        break
    fi

    # Telegram status every 30 min
    UPTIME=$(uptime)
    send_telegram "📈 *MCMC Status*
${LAST}
${UPTIME}"

    # Git checkpoint push every 3h (6 cycles)
    if [ $((CYCLES % 6)) -eq 0 ]; then
        push_checkpoint
        send_telegram "💾 Checkpoint auf GitHub gepusht."
    fi
done
