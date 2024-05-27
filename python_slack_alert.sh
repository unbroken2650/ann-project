#!/bin/bash

# 슬랙 알림 설정
CHANNEL="server-alert"
USERNAME="ssu_super"
TITLE="Alert!"
HOOK=https://hooks.slack.com/services/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

# Jupyter Notebook 실행 파일 설정
FILE="alert_test.py"

# 현재 시간을 파일명에 포함하도록 설정
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="execution_output_${TIMESTAMP}.log"

# 현재 시간과 실행한 파일명을 로그 파일에 기록
echo "실행 시간: $(date)" >$LOG_FILE
echo "실행한 파일: $FILE" >>$LOG_FILE

# Python 스크립트 실행 및 출력 저장
python $FILE >>$LOG_FILE 2>&1
# jupyter nbconvert --to notebook --execute --inplace --output $FILE $FILE > $LOG_FILE 2>&1

# 명령어의 종료 상태를 확인합니다
if [ $? -eq 0 ]; then
    MSG="✅성공: $FILE 실행이 완료되었습니다. \n로그 파일: $LOG_FILE\n실행 시간: $(date)\n실행한 파일: $FILE"
else
    MSG="⚠️에러: $FILE 실행 중 문제가 발생했습니다. 로그 파일을 확인하세요: $LOG_FILE\n실행 시간: $(date)\n실행한 파일: $FILE"
fi

# 슬랙으로 알림 전송
PAYLOAD="payload={\"channel\": \"$CHANNEL\", \"username\": \"$USERNAME\", \"text\": \"$TITLE \n\n $MSG\"}"
/usr/bin/curl -X POST --data-urlencode "$PAYLOAD" "$HOOK"
