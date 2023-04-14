if __name__ == '__main__':
    
    law_gpt_str = '1. O 2. X 3. O 4. O 5. O 6. X 7. X 8. O 9. X 10. X 11. O 12. O 13. X 14. O 15. X 16. O 17. X 18. X 19. X 20. X 21. X 22. X 23. X 24. O 25. O 26. O 27. X 28. X 29. X 30. O'
    law_gpt_answer = [s for s in law_gpt_str if s=='O' or s=='X']
    
    gpt4_answer = 'OXOOXOXXOOOOOOOXOOXXXOXXXOXOXO'

    correct = list('XXXOOOXXXXOXXXOXOOXXXXXOXOOXXX')

    law_gpt_score, gpt4_score = 0, 0
    for i, c in enumerate(correct):
        if c == law_gpt_answer[i]:
            law_gpt_score += 1
        if c == gpt4_answer[i]:
            gpt4_score += 1

    print(f'Law GPT Score = {law_gpt_score}/30')   
    print(f'GPT-4 Score = {gpt4_score}/30')