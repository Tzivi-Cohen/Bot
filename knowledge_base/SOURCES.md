# מקורות ואמינות – חומרי הלימוד ב־knowledge_base

## איך נוצרו החומרים

24 קובצי הלימוד בתיקיות הנושא נכתבו **במקור, במיוחד לפרויקט הזה**:
- הם נכתבו על ידי עוזר AI (Claude), לבקשת בעלת הפרויקט, על בסיס ידע מקצועי כללי.
- הם **לא** נלקחו או הועתקו מספר, מחוברת, מאתר או ממקור אחר.
- חוברת הלימוד של המורה לא שימשה כמקור.

הקישורים בהמשך הם מקורות שבהם **בדקנו** פרטים מסוימים אחרי הכתיבה. הם לא המקור של הטקסט.

> חומר שנכתב בעזרת AI עלול להכיל טעויות. לפני שימוש לצורך למידה או עבודה, כדאי לאמת אותו מול התיעוד הרשמי.

## 1. פרטים שנבדקו מול מקורות רשמיים (10/11/2026)

### Arduino

המקור: קובצי ה־Language Reference הרשמי של Arduino ב־GitHub.
<https://github.com/arduino/reference-en>

| קובץ במקור | מה נבדק |
|---|---|
| `Language/Functions/Analog IO/analogWrite.adoc` | פיני PWM ב־UNO (3, 5, 6, 9, 10, 11). תדר של כ־490Hz, ובפינים 5 ו־6 כ־980Hz. טווח 0–255. ההערה ש־Duty Cycle בפינים 5 ו־6 גבוה מהצפוי |
| `Language/Functions/Analog IO/analogRead.adoc` | ADC של 10 ביט (0–1023) ב־UNO, וכ־100 מיקרו־שניות לקריאה |
| `Language/Functions/Time/millis.adoc` | חזרה לאפס אחרי כ־50 יום, והטיפוס `unsigned long` |
| `Language/Functions/External Interrupts/attachInterrupt.adoc` | פינים 2 ו־3 לפסיקות ב־UNO. המצבים LOW, ‏CHANGE, ‏RISING, ‏FALLING. ‏`volatile`, ומה קורה ל־`delay()` ול־`millis()` בתוך ISR |
| `Language/Functions/Communication/Serial/setTimeout.adoc` | Timeout ברירת מחדל של 1000ms |
| `Language/Functions/Communication/Serial/print.adoc` | ברירת מחדל של שתי ספרות אחרי הנקודה ב־float |

### STM32 HAL

המקור: קוד המקור הרשמי של ST לדרייברי HAL של STM32F4.
<https://github.com/STMicroelectronics/stm32f4xx-hal-driver>

| קובץ במקור | מה נבדק |
|---|---|
| `Inc/stm32f4xx_hal_gpio.h` | חתימות `HAL_GPIO_ReadPin`, ‏`WritePin`, ‏`TogglePin`, ‏`EXTI_IRQHandler`, ‏`EXTI_Callback`. הערכים `GPIO_PIN_RESET` ו־`GPIO_PIN_SET`. מצבי `GPIO_MODE_*` והגדרות Pull |
| `Inc/stm32f4xx_hal_uart.h` | חתימות Transmit ו־Receive, הגרסאות `_IT`, וה־Callbacks |
| `Inc/stm32f4xx_hal_spi.h` | חתימות Transmit, ‏Receive ו־TransmitReceive |
| `Inc/stm32f4xx_hal_i2c.h` | חתימות Master_Transmit, ‏Master_Receive, ‏Mem_Read, ‏Mem_Write, ‏IsDeviceReady, והמאקרו `I2C_MEMADD_SIZE_8BIT` |
| `Src/stm32f4xx_hal_i2c.c` | הדרישה להזיז את הכתובת בת 7 הביטים שמאלה לפני הקריאה לפונקציה |
| `Inc/stm32f4xx_hal_tim.h` | חתימות `HAL_TIM_Base_Start_IT`, ‏`HAL_TIM_PWM_Start`, ‏`HAL_TIM_PeriodElapsedCallback`, והמאקרואים `__HAL_TIM_SET_COMPARE` ו־`__HAL_TIM_GET_COUNTER` |
| `Inc/stm32f4xx_hal_adc.h` | חתימות ADC, ו־`ADC_RESOLUTION_12B/10B/8B/6B` |

## 2. בדיקה בהרצה בפועל (10/11/2026)

- **Python:** דוגמאות הקוד עם פלט צפוי, בקובצי Python ובקובצי Data-Structures-and-Algorithms, הורצו ב־Python 3.13. הפלטים בטקסט תואמים להרצה.
  - **טעות שנמצאה ותוקנה:** מספר התנועות במשפט "Embedded Systems" הוא 4, ולא 5.
  - **תיקון נוסף:** ההסבר על האופרטור `//` הורחב, כך שיתאר גם את התוצאה כשאחד האופרנדים float (‏`7.0 // 2 == 3.0`). הדוגמאות החדשות נבדקו בהרצה.
- **C:** הדוגמאות מקובצי C, מבני נתונים ו־Embedded-Systems קומפלו והורצו ב־MSVC 2022 ‏(`/W4 /std:c11`), בלי אזהרות. הפלטים תואמים לטקסט.
- **C++:** הדוגמאות קומפלו והורצו ב־MSVC 2022 ‏(`/W4 /std:c++17`), בלי אזהרות. הפלטים תואמים לטקסט.
- **חישובים מספריים:** חוק אוהם, הספק, מחלקי מתח, RC, נוסחת הטיימר, הזזת כתובת I2C ו־Checksum נבדקו בחישוב חוזר.

## 3. מה **לא** אומת מול מקור חיצוני

התוכן הבא נכתב מידע מקצועי כללי, ולא נבדק מול מסמך ספציפי:

| נושא | מה לא נבדק |
|---|---|
| Electronics | כל הקבצים: ערכים טיפוסיים של Vf ו־β, סדרת E12, קוד הצבעים, ‏63.2% ב־τ |
| ATmega328P | מגבלת הזרם לפין |
| TMP36 | נתוני החיישן |
| Arduino | גודל Buffer הקבלה של Serial |
| לוחות Nucleo | פיני LD2 ו־B1. מסומנים בטקסט כ"תלוי לוח" |
| STM32 | פרטי NVIC ו־EXTI לפי משפחה, כפל שעון הטיימר, ונוסחת f_update |
| Embedded Systems | מושגי RTOS, ‏CAN ומפת הזיכרון |
| C++ | ההערה על `std::map` |
| Python | Timsort |

## 4. היכן לאמת

אלה אתרי תיעוד רשמיים שמומלץ לבדוק בהם. הם **לא** שימשו לכתיבת הטקסט.

| נושא | אתר |
|---|---|
| Python | <https://docs.python.org/3/> |
| C ו־C++ | <https://en.cppreference.com/> |
| Arduino | <https://docs.arduino.cc/language-reference/> |
| STM32 | <https://www.st.com/> – ה־Reference Manual, ה־Datasheet ומדריך ה־HAL של הדגם |
| FreeRTOS | <https://www.freertos.org/> |
