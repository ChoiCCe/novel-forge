# -*- coding: utf-8 -*-
"""通用多角色有声书前置处理。

保留模块：
    corpus       正文章节切分、多引号抽取、人名挖掘、台词归因、性别/年龄证据
    name_vote    人名多模型投票
    llm_resolve  待确定引号/说话人的 LLM 归属
    voices       edge 声线池探测与音色距离（只做测量，不参与分配）
    voice_sources 声线准入与池子维护（默认不做音频级判重）

旧的 engine / profile / cli.run（滑动窗口分配、章边界换色、回收窗口曲线）
已删除；声线分配以 outputs/novel-forge/v17_cast_official/ 为准。
"""
