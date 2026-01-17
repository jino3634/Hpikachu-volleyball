// 중앙 관리용 shaping 계수 정의
// 여기 값만 바꾸면 전체 shaping 동작이 바뀜

export const SHAPING = {
//   SERVE_CROSS_NET: 0.30,      // 서브 성공 (최초 네트 통과)
//   RETURN_CROSS_NET: 0.30,     // 리턴 성공 (내 코트 관측 후 네트 통과)
//   DEFENSIVE_SAVE: 0.20,       // 수비 세이브
//   TERMINAL_FAIL: 0.10,        // 터미널 실패 패널티 (감점)
  SERVE_CROSS_NET: 0,      // 서브 성공 (최초 네트 통과)
  RETURN_CROSS_NET: 0,     // 리턴 성공 (내 코트 관측 후 네트 통과)
  DEFENSIVE_SAVE: 0,       // 수비 세이브
  TERMINAL_FAIL: 0,        // 터미널 실패 패널티 (감점)
};
