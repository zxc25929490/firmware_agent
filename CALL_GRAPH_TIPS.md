Firmware Call Graph 查詢筆記
1️⃣ 上下層方向概念

在 firmware call graph 裡，有兩種搜尋方向：

🔽 DOWN（往下層 / callee）

代表：

這個 function 呼叫了誰？

例如：

Service_CPU_Thermal_Table
    → cpu_fan_update
        → fan_update_window

使用方式：

python query_paths.py --func Service_CPU_Thermal_Table
🔼 UP（往上層 / caller）

代表：

誰呼叫了這個 function？

例如你查到：

Service_CPU_Thermal_Table
    → Hook_Timer100msEventB
        → Timer100msEventB
            → service_1mS
                → main_service
                    → main

這代表：

main
 └─ main_service
     └─ service_1mS
         └─ Timer100msEventB
             └─ Hook_Timer100msEventB
                 └─ Service_CPU_Thermal_Table

使用方式：

python query_paths.py --func Service_CPU_Thermal_Table --up
🧠 2️⃣ 如何解讀 UP 結果？

當你看到：

Service_CPU_Thermal_Table
    → Hook_Timer100msEventB
        → Timer100msEventB
            → service_1mS
                → main_service
                    → main

代表：

它是被 timer 週期性呼叫

不是 event-driven

不是中斷直接進來

是從 main loop 進入

這對 debug 非常重要。