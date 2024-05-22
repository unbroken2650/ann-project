#!/bin/bash

# 슬랙 알림 설정
CHANNEL="server-alert"
USERNAME="ssu_super"
EMOJI="⚠️"
TITLE="Alert!"
HOOK=https://hooks.slack.com/services/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

# Jupyter Notebook 실행
NOTEBOOK="main.ipynb"
OUTPUT_NOTEBOOK="output_notebook_$NOTEBOOK.ipynb"
LOG_FILE="execution_output.log"
jupyter nbconvert --execute --to notebook --output "$OUTPUT_NOTEBOOK" "$NOTEBOOK" &> "$LOG_FILE"

# 명령어의 종료 상태를 확인합니다
if [ $? -eq 0 ]; then
    MSG="성공: $NOTEBOOK 실행이 완료되었습니다. 결과 파일: $OUTPUT_NOTEBOOK\n로그 파일: $LOG_FILE"
else
    MSG="에러: $NOTEBOOK 실행 중 문제가 발생했습니다. 로그 파일을 확인하세요: $LOG_FILE"
fi

# 슬랙으로 알림 전송
PAYLOAD="payload={\"channel\": \"$CHANNEL\", \"username\": \"$USERNAME\", \"text\": \"$TITLE \n\n $MSG\", \"icon_emoji\": \"$EMOJI\"}"
/usr/bin/curl -X POST --data-urlencode "$PAYLOAD" "$HOOK"
