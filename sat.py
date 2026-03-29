import georinex as gr

nav_file = 'samsung_nav.nav.rnx'

try:
    # אנחנו אומרים לו לטעון רק GPS (G) ו-Galileo (E)
    # זה עוקף את השגיאה של מערכת J (היפנית)
    nav = gr.load(nav_file, use=['G', 'E'])

    print("--- ✅ קובץ הניווט סונן ונטען בהצלחה! ---")
    print(f"מערכות בשימוש: {list(nav.sv.values)[:5]}...")

    # בדיקה קטנה לראות שיש לנו נתונים
    first_sv = nav.sv.values[0]
    print(f"נתונים זמינים עבור לוויין: {first_sv}")

except Exception as e:
    print(f"❌ עדיין יש שגיאה: {e}")
    print("\nניסיון אחרון - בוא נטען רק GPS:")
    try:
        nav = gr.load(nav_file, use='G')
        print("✅ טעינת GPS בלבד הצליחה!")
    except Exception as e2:
        print(f"❌ גם זה נכשל: {e2}")