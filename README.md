# 基于python的学生心理健康数据分析与可视化系统需求文档

## 1. 系统概述

### 1.1 系统名称
学生心理健康数据分析与可视化系统

### 1.2 系统目标
本系统作为毕业设计项目，旨在为学校提供学生心理健康数据分析与可视化解决方案，聚焦于**心理测试数据的深度分析和直观可视化**，辅助学校发现学生心理健康规律，为心理健康教育提供数据支持。系统将实现基础的心理测试、数据管理功能，并添加**机器学习预测、交互式可视化**等AI特色功能，展示数据分析与可视化技术在心理健康领域的应用。

### 1.3 系统架构
- **前端**：Bootstrap 5 + ECharts 5 + Jinja2模板 + 原生JavaScript
- **后端**：Flask 2.3 + SQLAlchemy 2.0
- **数据库**：SQLite（开发环境/毕设演示）/ MySQL（可选）
- **AI模块**：scikit-learn + XGBoost + Hugging Face Transformers（可选）
- **数据来源**：模拟数据 + Kaggle公开数据集

### 1.4 数据库设计

#### 1.4.1 表关系图

```
+----------------+     +------------------+
|     User       |     |  TestSession     |
+----------------+     +------------------+
| id (PK)        |-----| id (PK)          |
| username       | 1  N| user_id (FK)     |
| password_hash  |     | test_type        |
| role           |     | current_question |
| status         |     | total_questions  |
| created_at     |     | answers          |
| class_name     |     | session_status   |
| student_id     |     | created_at       |
+----------------+     | updated_at       |
          |            | expires_at       |
          |            | device_info      |
          |            +------------------+
          |
          |            +------------------+
          |            |  TestResult      |
          |            +------------------+
          | 1        N | id (PK)          |
          +------------| user_id (FK)     |
                       | test_type        |
                       | score            |
                       | test_date        |
                       | test_result      |
                       +------------------+
                              |
                              | 1        N
                              +------------+------------------+
                                           |  MLPredictionResult |
                                           +------------------+
                                           | id (PK)          |
                                           | user_id (FK)     |
                                           | prediction_type  |
                                           | prediction_value |
                                           | prediction_date  |
                                           | model_version    |
                                           | feature_data     |
                                           +------------------+
                              |
                              |
                              | 1        N
                              +------------+------------------+
                                           |  Conversation    |
                                           +------------------+
                                           | id (PK)          |
                                           | user_id (FK)     |
                                           | user_message     |
                                           | bot_response     |
                                           | emotion          |
                                           | timestamp        |
                                           +------------------+
```

#### 1.4.2 索引设计

为了优化系统性能，针对高频查询字段添加以下索引：

```sql
-- 用户表索引
CREATE INDEX idx_user_username ON user(username);
CREATE INDEX idx_user_role ON user(role);
CREATE INDEX idx_user_status ON user(status);

-- 测试结果表索引
CREATE INDEX idx_test_result_user_id ON test_result(user_id);
CREATE INDEX idx_test_result_date ON test_result(test_date);
CREATE INDEX idx_test_result_type ON test_result(test_type);
CREATE INDEX idx_test_result_user_type ON test_result(user_id, test_type);

-- 机器学习预测结果表索引
CREATE INDEX idx_ml_prediction_user_id ON ml_prediction_result(user_id);
CREATE INDEX idx_ml_prediction_type ON ml_prediction_result(prediction_type);
CREATE INDEX idx_ml_prediction_date ON ml_prediction_result(prediction_date);

-- 会话表索引
CREATE INDEX idx_test_session_user_id ON test_session(user_id);
CREATE INDEX idx_test_session_status ON test_session(session_status);
CREATE INDEX idx_test_session_expires ON test_session(expires_at);

-- 对话表索引
CREATE INDEX idx_conversation_user_id ON conversation(user_id);
CREATE INDEX idx_conversation_timestamp ON conversation(timestamp);
CREATE INDEX idx_conversation_emotion ON conversation(emotion);

-- 日志表索引
CREATE INDEX idx_data_access_user_id ON data_access_log(user_id);
CREATE INDEX idx_data_access_resource ON data_access_log(resource_type, resource_id);
CREATE INDEX idx_data_access_time ON data_access_log(accessed_at);

CREATE INDEX idx_system_operation_operator ON system_operation_log(operator_id, operator_type);
CREATE INDEX idx_system_operation_time ON system_operation_log(executed_at);
```

#### 1.4.3 数据量预估

结合毕设演示需求，系统模拟数据规模如下：

| 表名 | 预计数据量 | 说明 |
|------|------------|------|
| user | 1,000条 | 模拟1000名学生用户和10名管理员用户 |
| test_result | 5,000条 | 平均每位学生完成5次心理测试 |
| ml_prediction_result | 10,000条 | 每次测试生成2个预测结果（焦虑、抑郁） |
| test_session | 2,000条 | 包含活跃会话和历史会话 |
| conversation | 50,000条 | 平均每位学生50条对话记录 |
| data_access_log | 100,000条 | 系统操作日志 |
| system_operation_log | 1,000条 | 系统管理日志 |

该数据规模能够有效演示系统的功能和性能，同时保持系统在开发环境下的良好运行效率。

### 1.5 Kaggle数据集处理

#### 1.5.1 数据集选择
- **选择依据**：选择与大学生心理健康相关的数据集，包括心理测试得分、压力源、焦虑抑郁症状等维度
- **推荐数据集**：
  - Student Mental Health Dataset（学生心理健康数据集）
  - Anxiety and Depression Dataset（焦虑抑郁数据集）
  - Stress Level Detection Dataset（压力水平检测数据集）
  - College Student Mental Health（大学生心理健康数据集）

#### 1.5.2 数据集爬取（Python实现）
- **爬取工具**：使用Python requests + BeautifulSoup或Kaggle API进行数据爬取
- **爬取流程**：
  1. 配置Kaggle API密钥（~/.kaggle/kaggle.json）
  2. 使用Python Kaggle API下载目标数据集
  3. 自动解压缩数据集文件
  4. 验证数据集完整性
- **实现代码示例**：
  ```python
  import kaggle
  import zipfile
  import os
  
  # 配置Kaggle API
  kaggle.api.authenticate()
  
  # 下载数据集
  dataset_name = "dataset-owner/dataset-name"
  output_dir = "data/raw"
  
  # 确保输出目录存在
  os.makedirs(output_dir, exist_ok=True)
  
  try:
      # 下载数据集并解压缩
      kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
      
      # 验证数据集
      if os.path.exists(f"{output_dir}/data.csv"):
          print("数据集下载成功")
      else:
          print("数据集下载完成，但未找到预期的数据文件")
  except kaggle.KaggleApiError as e:
      print(f"Kaggle API错误: {str(e)}")
      print("请检查API密钥配置和数据集名称是否正确")
  except Exception as e:
      print(f"数据集下载失败: {str(e)}")
      print("可能的原因包括网络问题、权限不足或存储空间不足")
  
  # 可选：添加重试机制
  max_retries = 3
  retry_count = 0
  
  while retry_count < max_retries:
      try:
          kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
          print("数据集下载成功")
          break
      except Exception as e:
          retry_count += 1
          print(f"下载失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
          if retry_count < max_retries:
              import time
              time.sleep(5)  # 等待5秒后重试
  
  if retry_count == max_retries:
      print("已达到最大重试次数，下载失败")
  ```

#### 1.5.3 数据清洗与预处理（Python实现）
- **清洗目标**：确保数据质量，去除噪声和异常值，统一数据格式
- **清洗步骤**：
  1. **数据读取**：使用Python pandas读取CSV数据
  2. **缺失值处理**：
     - 数值型数据：使用均值/中位数填充
     - 分类型数据：使用众数填充或删除缺失值
  3. **异常值处理**：
     - 使用箱线图（IQR）检测异常值
     - 对极端异常值进行删除或修正
  4. **数据格式统一**：
     - 统一日期格式
     - 统一编码格式
     - 标准化数值型数据
  5. **特征工程**：
     - 提取有用特征
     - 编码分类型数据
     - 特征缩放（归一化/标准化）
  6. **数据验证**：验证数据完整性和一致性

#### 1.5.4 数据集成与映射
- **集成目标**：将Kaggle数据集与系统内部数据模型进行映射和集成
- **集成步骤**：
  1. **数据字段映射**：
     - 提前创建详细的字段映射表
     - 将Kaggle数据集中的字段映射到系统数据模型
     - 例如：将"stress_level"映射到"standard_score"
     - 预留"空值"分支逻辑，处理字段不匹配情况
  2. **数据格式转换**：
     - 将Kaggle数据转换为系统所需格式
     - 例如：将文本标签转换为数值编码
     - 处理缺失值和异常值
     - **时间维度对齐**：
       - 标准化时间格式：将Kaggle数据的时间字段统一转换为ISO 8601格式（YYYY-MM-DD HH:MM:SS）
       - 时间范围映射：将Kaggle数据的时间范围映射到系统的时间维度（如将历史数据按季度/月份与系统测试时间对齐）
       - 时间粒度统一：根据Kaggle数据的时间粒度（日/周/月），与系统内部测试记录的时间粒度进行统一
       - 缺失时间处理：对缺少时间信息的Kaggle数据，根据数据特征进行合理推断或标记
       - 时间戳验证：确保转换后的时间戳在系统支持的合理范围内（如2020-2025年）
       
       实现代码示例：
       ```python
       import pandas as pd
       from datetime import datetime
       
       # 读取Kaggle数据
       kaggle_data = pd.read_csv('kaggle_student_mental_health.csv')
       
       # 标准化时间格式（假设Kaggle数据有date字段）
       if 'date' in kaggle_data.columns:
           # 转换各种可能的时间格式
           kaggle_data['standardized_date'] = pd.to_datetime(kaggle_data['date'], 
                                                              format='mixed', 
                                                              errors='coerce')
           # 将时间转换为ISO 8601格式
           kaggle_data['standardized_date'] = kaggle_data['standardized_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
           
       # 处理缺少时间信息的数据
       kaggle_data['standardized_date'] = kaggle_data['standardized_date'].fillna('1900-01-01 00:00:00')  # 标记缺失时间
       ```
  3. **数据导入**：
     - 使用Python SQLAlchemy将清洗后的数据导入系统数据库
     - 支持批量导入和增量导入
     - 实现数据导入日志记录
  4. **数据验证**：
     - 验证导入数据的完整性
     - 检查数据关系完整性
     - 生成数据质量报告

#### 1.5.5 数据更新机制
- **定期更新**：设置定时任务，定期从Kaggle更新最新数据集
- **增量更新**：只更新新增或变化的数据
- **更新日志**：记录数据更新时间、来源和内容变化

#### 1.5.6 数据隐私保护
- **数据脱敏**：对敏感字段进行脱敏处理
  - **需要脱敏的字段**：
    - 姓名：使用姓氏首字母+星号替代（如"李**"）
    - 学号：隐藏中间4位（如"2020****001"）
    - 身份证号：隐藏中间8位（如"110101********1234"）
    - 手机号：隐藏中间4位（如"138****1234"）
    - 邮箱：隐藏用户名部分（如"*@example.com"）
    - 家庭地址：保留省市，隐藏详细地址（如"北京市朝阳区***"）
    - IP地址：隐藏最后一段（如"192.168.1.*"）
  - **脱敏方式**：
    - 替换法：使用星号、首字母等替换敏感信息
    - 截断法：截取部分信息，隐藏其余部分
    - 加密法：对核心敏感信息进行单向加密
    - 哈希法：对需要关联但不直接展示的信息使用哈希处理
  - **脱敏规则实现代码示例**：
    ```python
    import hashlib
    import re
    
    def desensitize_data(data):
        """
        对数据进行脱敏处理
        
        参数:
            data: 包含敏感信息的字典
        
        返回:
            脱敏后的数据字典
        """
        desensitized_data = data.copy()
        
        # 姓名脱敏
        if 'name' in data:
            name = data['name']
            if len(name) > 1:
                desensitized_data['name'] = name[0] + '*' * (len(name) - 1)
        
        # 学号脱敏 - 适配不同长度学号
        if 'student_id' in data:
            student_id = str(data['student_id'])
            length = len(student_id)
            if length <= 4:
                # 长度不足时只保留第一个字符
                desensitized_data['student_id'] = student_id[0] + '*' * (length - 1)
            elif length <= 6:
                # 中等长度保留前后部分
                desensitized_data['student_id'] = student_id[:2] + '*' * (length - 3) + student_id[-1:]
            elif length <= 10:
                # 大学生学号（通常为8-10位）：保留前3位和后3位，中间用*填充
                middle = length - 6
                desensitized_data['student_id'] = student_id[:3] + '*' * middle + student_id[-3:]
            else:
                # 超长学号保留前4位和后4位，中间用*填充
                middle = length - 8
                desensitized_data['student_id'] = student_id[:4] + '*' * middle + student_id[-4:]
        
        # 手机号脱敏
        if 'phone' in data:
            phone = str(data['phone'])
            if re.match(r'^1[3-9]\d{9}$', phone):
                desensitized_data['phone'] = phone[:3] + '****' + phone[-4:]
        
        # 邮箱脱敏 - 保留用户名首字母以便区分同一域名下的不同用户
        if 'email' in data:
            email = data['email']
            if '@' in email:
                username, domain = email.split('@', 1)
                if len(username) > 1:
                    # 保留用户名首字母
                    desensitized_data['email'] = username[0] + '*' * (len(username) - 1) + '@' + domain
                else:
                    # 用户名只有一个字符时直接使用
                    desensitized_data['email'] = username + '@' + domain
        
        # 身份证号脱敏
        if 'id_card' in data:
            id_card = str(data['id_card'])
            if len(id_card) == 18:
                desensitized_data['id_card'] = id_card[:6] + '********' + id_card[-4:]
        
        return desensitized_data
    ```
    
- **匿名化处理**：去除个人身份信息
  - 为每个学生生成唯一匿名ID，替代真实身份标识
  - 去除与个人身份直接相关的字段（如真实姓名、身份证号等）
  - 对群体数据进行聚合处理，确保无法通过组合信息识别个人
- **数据访问控制**：严格控制数据集的访问权限
  - 基于RBAC模型实现细粒度的访问控制
  - 敏感数据仅允许授权人员访问
  - 数据导出需经过审批并记录日志
- **遵守数据使用协议**：确保符合Kaggle数据集的使用条款
  - 明确数据用途范围
  - 不将数据用于商业目的
  - 适当引用数据来源

## 2. 功能需求

### 2.0 功能模块交互流程

#### 2.0.1 用户-系统交互流程图

##### 登录流程
1. 用户访问系统首页
2. 点击"登录"按钮，跳转到登录页面
3. 用户输入用户名和密码
4. 前端验证输入格式（非空、格式正确）
5. 前端发送POST请求到`/api/auth/login`
6. 后端验证用户名和密码
7. 验证成功：生成JWT令牌，返回200 OK和用户信息
8. 验证失败：返回401 Unauthorized和错误信息
9. 前端保存令牌到localStorage，跳转到对应角色首页

##### 测试提交流程
1. 用户从测试列表页面选择测试类型
2. 跳转到测试说明页面，用户确认开始测试
3. 进入答题页面，用户逐题作答
4. 每答完一题，前端自动保存到localStorage
5. 每答完5题，前端向后端发送保存请求到`/api/test/save_progress`
6. 用户完成所有题目，点击"提交"按钮
7. 前端验证所有题目已作答
8. 前端发送POST请求到`/api/test/submit`
9. 后端计算得分，生成测试结果
10. 后端调用机器学习模型进行预测
11. 后端保存测试结果和预测结果到数据库
12. 后端返回200 OK和测试结果ID
13. 前端跳转到结果页面

##### 报告生成流程
1. 用户从个人中心或测试结果页面点击"查看报告"
2. 前端发送GET请求到`/api/report/generate?test_result_id=xxx`
3. 后端查询测试结果和预测数据
4. 后端生成个性化报告内容
5. 后端返回200 OK和报告数据
6. 前端渲染报告页面，展示可视化图表
7. 用户可选择下载PDF报告

#### 2.0.2 前端页面跳转逻辑

| 页面名称 | 路由路径 | 跳转来源 | 跳转目标 | 权限要求 |
|---------|---------|---------|---------|---------|
| 首页 | / | 登录成功 | 测试列表、个人中心 | 学生/管理员 |
| 登录页 | /login | 未登录访问受保护页面 | 首页 | 未登录 |
| 测试列表页 | /tests | 首页 | 测试说明页 | 学生 |
| 测试说明页 | /tests/:type/info | 测试列表页 | 答题页 | 学生 |
| 答题页 | /tests/:type/answer | 测试说明页、断点续答 | 结果页 | 学生 |
| 结果页 | /results/:id | 答题页 | 报告页、测试历史页 | 学生 |
| 报告页 | /report/:id | 结果页、测试历史页 | 个人中心 | 学生 |
| 测试历史页 | /history | 个人中心 | 报告页 | 学生 |
| 个人中心页 | /profile | 首页 | 测试历史页、设置页 | 学生/管理员 |
| 统计分析页 | /admin/stats | 管理员首页 | 数据管理页 | 管理员 |
| 数据管理页 | /admin/data | 管理员首页 | 用户管理页 | 管理员 |
| 用户管理页 | /admin/users | 管理员首页 | 系统配置页 | 管理员 |
| 系统配置页 | /admin/config | 管理员首页 | 统计分析页 | 管理员 |

#### 2.0.3 前后端数据交互格式

##### API 请求 / 响应示例

###### 登录请求
```json
POST /api/auth/login
Request:
{
  "username": "student123",
  "password": "Password123!"
}

Response (200 OK):
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1001,
    "username": "student123",
    "role": "student",
    "class_name": "2022级计算机1班",
    "student_id": "20220101"
  },
  "expires_at": "2023-12-01 16:30:22"
}

Response (401 Unauthorized):
{
  "error": "Invalid username or password",
  "code": 401
}
```

###### 测试提交请求
```json
POST /api/test/submit
Request:
{
  "user_id": 1001,
  "test_type": "sas",
  "answers": [2, 3, 1, 4, 2, 3, 1, 4, 2, 3, 1, 4, 2, 3, 1, 4, 2, 3, 1, 4],  // 20题的答案数组
  "submit_time": "2023-12-01 15:30:22"
}

Response (200 OK):
{
  "test_result_id": 2001,
  "score": 55.5,
  "interpretation": "轻度焦虑",
  "prediction": {
    "anxiety_risk": 0.65,
    "depression_risk": 0.32
  },
  "redirect_url": "/results/2001"
}

Response (400 Bad Request):
{
  "error": "Invalid test data: missing answers",
  "code": 400
}
```

###### 报告生成请求
```json
GET /api/report/generate?test_result_id=2001
Response (200 OK):
{
  "report_id": 3001,
  "user_info": {
    "id": 1001,
    "username": "student123",
    "class_name": "2022级计算机1班"
  },
  "test_info": {
    "test_type": "sas",
    "test_date": "2023-12-01 15:30:22",
    "score": 55.5,
    "interpretation": "轻度焦虑"
  },
  "prediction_info": {
    "anxiety_risk": 0.65,
    "depression_risk": 0.32,
    "model_version": "v1.0"
  },
  "recommendations": [
    "您的焦虑风险中等，建议定期进行放松练习，如深呼吸、冥想等。",
    "保持规律的作息时间，多参加户外活动。"
  ],
  "visualization_data": {
    "radar_chart_data": [/* 雷达图数据 */],
    "line_chart_data": [/* 折线图数据 */]
  }
}
```

#### 2.0.4 异常场景处理

##### 网络中断处理
1. **前端处理**：
   - 显示网络连接失败提示
   - 提供重试按钮
   - 自动保存当前状态到localStorage
   - 网络恢复后自动提示用户

2. **后端处理**：
   - 实现请求超时机制（默认30秒）
   - 对于幂等操作（如测试提交），支持重复提交
   - 记录请求日志，便于问题排查

##### 数据校验失败处理
1. **前端校验**：
   - 实时验证用户输入格式
   - 显示友好的错误提示
   - 阻止无效数据提交

2. **后端校验**：
   - 再次验证所有请求数据
   - 返回详细的错误信息
   - 记录校验失败日志

##### 系统异常处理
1. **前端处理**：
   - 显示通用错误页面
   - 提供返回首页按钮
   - 收集错误信息并发送到后端

2. **后端处理**：
   - 捕获所有未处理异常
   - 返回500 Internal Server Error
   - 记录详细的错误堆栈信息
   - 实现告警机制（可选）

##### 权限不足处理
1. **前端处理**：
   - 隐藏无权限的菜单和按钮
   - 访问无权限页面时跳转到403页面

2. **后端处理**：
   - 验证所有请求的权限
   - 返回403 Forbidden
   - 记录权限错误日志

### 2.1 用户认证与权限管理

#### 2.1.1 登录与注销
- **登录流程**：详见2.0.1节用户-系统交互流程图中的登录流程
- **注销流程**：
  - 销毁会话数据
  - 清除会话cookie
  - 记录注销时间
- **密码安全**：
  - 使用bcrypt算法加密存储
  - 密码长度至少8位
  - 包含字母、数字和特殊字符

#### 2.1.2 用户角色与权限
| role | permissions | accessible_pages |
|------|------|------------|
| **学生** | 进行心理测试、查看测试历史、查看个人分析报告 | 首页、心理测试、测试历史、个人分析 |
| **管理员** | 查看统计分析、管理用户和测试数据、查看系统日志、配置系统参数、查看班级心理状态、管理预警信息 | 首页、统计分析、数据管理、用户管理、系统配置、班级分析、预警管理 |

#### 2.1.3 RBAC权限控制（详细版）
- **基于角色的访问控制（RBAC）**：采用经典的RBAC模型，实现用户-角色-权限的三级授权体系
- **权限类型**：
  - 功能权限：控制用户可访问的页面和操作
  - 数据权限：控制用户可访问的数据范围（如学生只能访问自己的数据）
- **角色定义**：
  | role_id | role_name | role_description |
  |--------|----------|----------|
  | R001   | 学生     | 系统的主要使用群体，进行心理测试和咨询 |
  | R002   | 管理员   | 系统维护和配置人员，管理所有用户和权限 |
- **数据权限划分标准**：
  | role | data_access_scope | data_operation_permissions |
  |------|----------------|--------------|
  | 学生 | 个人测试数据、个人咨询记录、个人预警信息 | 查看、创建 |
  | 管理员 | 所有系统数据 | 查看、编辑、导出、删除、系统配置 |
- **数据权限实现方式**：
  - 基于用户ID的行级权限控制：学生只能访问user_id与自身ID匹配的数据
  - 基于角色的权限控制：不同角色对应不同的数据访问视图
- **权限实现**：
  - 使用Flask装饰器实现权限验证
  - 权限信息存储在数据库中，支持动态分配
  - 实现细粒度的权限控制，如"查看测试结果"、"修改预警状态"等
  - 按钮级权限：毕设阶段暂不实现
- **权限管理界面**：管理员可通过可视化界面管理角色和权限
- **权限验证流程**：
  1. 用户登录后获取角色信息
  2. 访问页面或执行操作时，装饰器检查用户角色权限
  3. 验证通过则允许访问，否则返回权限不足提示

#### 2.1.4 用户管理功能
- **用户列表**：
  - 分页展示所有用户
  - 支持按角色、状态筛选
  - 支持用户名搜索
- **用户操作**：
  - 创建用户
  - 编辑用户信息
  - 重置密码
  - 启用/禁用用户

### 2.2 心理测试功能

#### 2.2.1 测试类型与内容

##### 2.2.1.1 SCL-90（90项症状清单）
- **测试目的**：评估个体近一周内的心理症状严重程度
- **测试项目**：90个项目，涵盖10个因子
- **评分方法**：5级评分（1-无，2-轻度，3-中度，4-偏重，5-严重）
- **2022级大学生适配**：重点关注人际关系敏感、抑郁、焦虑、敌对、恐怖、偏执等因子，这些因子与当代大学生的心理健康状况高度相关

##### 2.2.1.2 SAS（焦虑自评量表）
- **测试目的**：评估个体近一周内的焦虑程度
- **测试项目**：20个项目
- **评分方法**：4级评分
- **结果解释**：
  - 标准分<50分：正常
  - 50-59分：轻度焦虑
  - 60-69分：中度焦虑
  - ≥70分：重度焦虑
- **2022级大学生适配**：增加对就业焦虑、考研焦虑、社交焦虑等现代大学生常见焦虑类型的评估维度

##### 2.2.1.3 SDS（抑郁自评量表）
- **测试目的**：评估个体近一周内的抑郁程度
- **测试项目**：20个项目
- **评分方法**：4级评分
- **结果解释**：
  - 标准分<53分：正常
  - 53-62分：轻度抑郁
  - 63-72分：中度抑郁
  - ≥73分：重度抑郁
- **2022级大学生适配**：增加对学业压力、社交隔离、未来迷茫等现代大学生常见抑郁诱因的评估

##### 2.2.1.4 压力源评估量表（针对2022级大学生）
- **测试目的**：评估2022级大学生面临的主要压力源
- **测试项目**：30个项目，涵盖以下维度：
  - 学业压力（课程难度、考试压力、考研/就业压力）
  - 社交压力（人际关系、社交焦虑、孤独感）
  - 生活压力（经济压力、生活适应、健康问题）
  - 疫情影响（线上学习适应性、疫情心理阴影）
  - 未来规划（职业发展、人生目标）
- **评分方法**：5级评分（1-无压力，2-轻度压力，3-中度压力，4-重度压力，5-极度压力）
- **结果解释**：
  - 总得分<60分：压力水平低
  - 60-89分：压力水平中等
  - 90-119分：压力水平高
  - ≥120分：压力水平极高

##### 2.2.1.5 疫情后心理健康恢复量表
- **测试目的**：评估2022级大学生在疫情后的心理健康恢复情况
- **测试项目**：20个项目，涵盖以下维度：
  - 情绪恢复（焦虑、抑郁情绪的恢复程度）
  - 学习适应（从线上学习到线下学习的适应情况）
  - 社交恢复（社交能力和社交意愿的恢复情况）
  - 心理韧性（面对困难和压力的心理承受能力）
- **评分方法**：4级评分（1-完全不符合，2-部分不符合，3-部分符合，4-完全符合）
- **结果解释**：
  - 总得分<40分：恢复情况较差
  - 40-55分：恢复情况一般
  - 56-70分：恢复情况良好
  - ≥71分：恢复情况优秀

#### 2.2.2 测试流程
- **详细流程**：详见2.0.1节用户-系统交互流程图中的测试提交流程
- **核心环节**：
  1. 测试前准备：展示说明、确认协议
  2. 测试进行：分页答题、实时保存
  3. 测试完成：自动计分、生成报告
  4. 结果处理：保存记录、调用模型预测

#### 2.2.2.1 测试进度保存机制
- **保存时机**：
  - 前端：使用localStorage每答完一题自动保存到本地
  - 后端：每答完5题自动保存到数据库，测试完成时最终保存
- **保存内容**：
  - 前端保存：测试ID、学生ID、测试类型、当前进度、已答题目答案、开始时间
  - 后端保存：测试会话记录、答题进度、当前答案状态、最后操作时间
- **未完成测试处理**：
  - 前端：localStorage永久保存，支持跨设备同步
  - 后端：创建test_session表保存未完成测试，支持跨设备恢复
  - 测试恢复逻辑：登录时检查未完成测试，提供"继续测试"选项
  - 数据一致性：前端和后端进度对比，以最新保存的进度为准
- **跨设备支持**：
  - 用户登录时同步未完成测试数据
  - 自动检测并合并不同设备的答题进度
  - 防止多设备同时答题的冲突处理
- **性能优化**：
  - 减少数据库写入次数，提高系统性能
  - 支持离线答题，无网络时仍可保存进度
  - 提交时进行数据验证，确保数据完整性
- **数据模型**：
  ```sql
  -- 测试会话表（保存未完成测试）
  CREATE TABLE test_session (
      id INTEGER PRIMARY KEY,
      user_id INTEGER NOT NULL,
      test_type VARCHAR(20) NOT NULL,
      current_question INTEGER DEFAULT 1,
      total_questions INTEGER NOT NULL,
      answers TEXT, -- JSON格式存储已答题目
      session_status ENUM('active', 'paused', 'expired') DEFAULT 'active',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      expires_at DATETIME, -- 30天后过期
      device_info VARCHAR(100), -- 设备标识
      FOREIGN KEY (user_id) REFERENCES user(id)
  );
  ```
  ```python
  # SQLAlchemy ORM定义
  from sqlalchemy import Column, Integer, String, Text, DateTime, Enum
  from sqlalchemy.ext.declarative import declarative_base
  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker
  from datetime import datetime
  
  Base = declarative_base()
  
  # 数据库连接配置（实际项目中应放在配置文件中）
  engine = create_engine('sqlite:///mental_health.db')
  SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
  
  class TestSession(Base):
      __tablename__ = 'test_session'
      
      id = Column(Integer, primary_key=True)
      user_id = Column(Integer, nullable=False)
      test_type = Column(String(20), nullable=False)
      current_question = Column(Integer, default=1)
      total_questions = Column(Integer, nullable=False)
      answers = Column(Text)  # JSON格式存储已答题目
      session_status = Column(Enum('active', 'paused', 'expired'), default='active')
      created_at = Column(DateTime, default=datetime.utcnow)
      updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
      expires_at = Column(DateTime)  # 30天后过期
      device_info = Column(String(100))  # 设备标识
  
  class Conversation(Base):
      __tablename__ = 'conversation'
      
      id = Column(Integer, primary_key=True, autoincrement=True)
      user_id = Column(Integer, nullable=False)
      user_message = Column(Text, nullable=False)
      bot_response = Column(Text, nullable=False)
      emotion = Column(String(50), nullable=False)
      timestamp = Column(DateTime, default=datetime.utcnow)

  # 数据访问日志表
  class DataAccessLog(Base):
      __tablename__ = 'data_access_log'
      
      id = Column(Integer, primary_key=True, autoincrement=True)
      user_id = Column(Integer, nullable=False)
      action_type = Column(String(50), nullable=False)  # 查询类型：user_query, admin_query, system_query
      resource_type = Column(String(50), nullable=False)  # 资源类型：conversation, test_result, user_info, system_setting
      resource_id = Column(Integer, nullable=True)  # 资源ID
      accessed_at = Column(DateTime, default=datetime.utcnow)
      ip_address = Column(String(45), nullable=True)  # IPv4/IPv6
      user_agent = Column(Text, nullable=True)  # 用户代理
      access_result = Column(String(20), nullable=False)  # 访问结果：success, failed, denied
      error_message = Column(Text, nullable=True)  # 错误信息
      accessed_fields = Column(Text, nullable=True)  # 访问的字段列表（JSON格式）
      operation_details = Column(Text, nullable=True)  # 操作详情（JSON格式）

  # 系统操作日志表
  class SystemOperationLog(Base):
      __tablename__ = 'system_operation_log'
      
      id = Column(Integer, primary_key=True, autoincrement=True)
      operator_id = Column(Integer, nullable=True)  # 操作用户ID，系统操作时为NULL
      operator_type = Column(String(20), nullable=False)  # 操作类型：admin, system, scheduled_task
      operation_name = Column(String(100), nullable=False)  # 操作名称
      operation_details = Column(Text, nullable=True)  # 操作详情（JSON格式）
      executed_at = Column(DateTime, default=datetime.utcnow)
      execution_result = Column(String(20), nullable=False)  # 执行结果：success, failed, partial
      error_stack_trace = Column(Text, nullable=True)  # 错误堆栈信息
      affected_records = Column(Integer, nullable=True)  # 影响的记录数

  # 用户表
  class User(Base):
      __tablename__ = 'user'
      
      id = Column(Integer, primary_key=True, autoincrement=True)
      username = Column(String(50), nullable=False, unique=True)
      password_hash = Column(String(255), nullable=False)
      role = Column(String(20), nullable=False)  # student, admin
      status = Column(String(20), nullable=False, default='active')  # active, inactive
      created_at = Column(DateTime, default=datetime.utcnow)
      class_name = Column(String(50))  # 班级信息
      student_id = Column(String(20), unique=True)  # 学号信息

  # 测试结果表
  class TestResult(Base):
      __tablename__ = 'test_result'
      
      id = Column(Integer, primary_key=True, autoincrement=True)
      user_id = Column(Integer, nullable=False)
      test_type = Column(String(50), nullable=False)  # sas, sds, etc.
      score = Column(Float, nullable=False)
      test_date = Column(DateTime, default=datetime.utcnow)
      test_result = Column(JSON, nullable=True)  # 完整测试结果（JSON格式）
      
      __table_args__ = (ForeignKeyConstraint(['user_id'], ['user.id']),)

  # 机器学习预测结果表
  class MLPredictionResult(Base):
      __tablename__ = 'ml_prediction_result'
      
      id = Column(Integer, primary_key=True, autoincrement=True)
      user_id = Column(Integer, nullable=False)
      prediction_type = Column(String(50), nullable=False)  # 预测类型：anxiety_risk, depression_risk, emotion_trend
      prediction_value = Column(Float, nullable=False)  # 预测值（0-1之间）
      prediction_date = Column(DateTime, default=datetime.utcnow)  # 预测日期
      model_version = Column(String(20), nullable=False)  # 模型版本
      feature_data = Column(Text, nullable=False)  # 用于预测的特征数据（JSON格式）
      
      __table_args__ = (ForeignKeyConstraint(['user_id'], ['user.id']),)

  # 创建所有表
Base.metadata.create_all(bind=engine)

# 日志记录工具函数
def log_data_access(user_id, action_type, resource_type, resource_id=None, ip_address=None,
                   user_agent=None, access_result='success', error_message=None,
                   accessed_fields=None, operation_details=None):
    """
    记录数据访问日志
    
    参数:
        user_id: 用户ID
        action_type: 查询类型（user_query, admin_query, system_query）
        resource_type: 资源类型（conversation, test_result, user_info, system_setting）
        resource_id: 资源ID（可选）
        ip_address: IP地址（可选）
        user_agent: 用户代理（可选）
        access_result: 访问结果（success, failed, denied）
        error_message: 错误信息（可选）
        accessed_fields: 访问的字段列表（JSON格式，可选）
        operation_details: 操作详情（JSON格式，可选）
    """
    session = SessionLocal()
    try:
        log_entry = DataAccessLog(
            user_id=user_id,
            action_type=action_type,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            access_result=access_result,
            error_message=error_message,
            accessed_fields=accessed_fields,
            operation_details=operation_details
        )
        session.add(log_entry)
        session.commit()
    except Exception as e:
        # 记录日志失败时，记录到系统日志
        log_system_operation(
            operator_type='system',
            operation_name='log_data_access_failure',
            operation_details=json.dumps({"error": str(e), "user_id": user_id, "resource_type": resource_type}),
            execution_result='failed'
        )
        session.rollback()
    finally:
        session.close()


def log_system_operation(operator_id=None, operator_type, operation_name, operation_details=None,
                         execution_result='success', error_stack_trace=None, affected_records=None):
    """
    记录系统操作日志
    
    参数:
        operator_id: 操作用户ID（系统操作时为None）
        operator_type: 操作类型（admin, system, scheduled_task）
        operation_name: 操作名称
        operation_details: 操作详情（JSON格式，可选）
        execution_result: 执行结果（success, failed, partial）
        error_stack_trace: 错误堆栈信息（可选）
        affected_records: 影响的记录数（可选）
    """
    session = SessionLocal()
    try:
        log_entry = SystemOperationLog(
            operator_id=operator_id,
            operator_type=operator_type,
            operation_name=operation_name,
            operation_details=operation_details,
            execution_result=execution_result,
            error_stack_trace=error_stack_trace,
            affected_records=affected_records
        )
        session.add(log_entry)
        session.commit()
    except Exception as e:
        # 系统日志记录失败时，记录到标准错误输出
        print(f"System log recording failed: {str(e)}")
        session.rollback()
    finally:
        session.close()


# 机器学习模型类（用于心理健康预测）
class MentalHealthMLModel:
    def __init__(self, model_type='random_forest'):
        """
        初始化机器学习模型
        
        参数:
            model_type: 模型类型（'random_forest' 或 'xgboost'）
        """
        self.model_type = model_type
        self.model = None
        self.model_version = "v1.0"
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        训练模型
        
        参数:
            X_train: 训练特征数据
            y_train: 训练标签数据
        """
        try:
            if self.model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif self.model_type == 'xgboost':
                from xgboost import XGBClassifier
                self.model = XGBClassifier(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # 记录模型训练事件
            log_system_operation(
                operator_type='system',
                operation_name='model_training',
                operation_details=json.dumps({"model_type": self.model_type, "model_version": self.model_version}),
                execution_result='success'
            )
            
            return True
        except Exception as e:
            log_system_operation(
                operator_type='system',
                operation_name='model_training_failure',
                operation_details=json.dumps({"model_type": self.model_type, "error": str(e)}),
                execution_result='failed'
            )
            return False
    
    def predict(self, X):
        """
        进行预测
        
        参数:
            X: 预测特征数据
            
        返回:
            预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        return self.model.predict_proba(X)[:, 1]  # 返回正类的概率
    
    def save_model(self, file_path):
        """
        保存模型到文件
        
        参数:
            file_path: 模型文件路径
        """
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, file_path):
        """
        从文件加载模型
        
        参数:
            file_path: 模型文件路径
            
        返回:
            加载的模型对象
        """
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)


# 个性化分析报告生成函数（与机器学习模型集成）
def generate_personalized_report(user_id):
    """
    生成个性化分析报告，集成机器学习预测结果
    
    参数:
        user_id: 用户ID
        
    返回:
        报告内容（字典格式）
    """
    session = SessionLocal()
    try:
        # 获取用户基本信息
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"用户ID {user_id} 不存在")
        
        # 获取用户的测试结果
        test_results = session.query(TestResult).filter(TestResult.user_id == user_id).order_by(TestResult.test_date.desc()).all()
        
        # 获取用户的情感分析记录
        emotion_records = session.query(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.timestamp.desc()).limit(50).all()
        
        # 加载机器学习模型进行预测
        ml_model = MentalHealthMLModel.load_model('models/mental_health_model.pkl')
        
        # 准备预测特征数据
        features = prepare_prediction_features(user_id)
        
        # 进行预测
        anxiety_prediction = ml_model.predict(features) * 100  # 焦虑风险（百分比）
        depression_prediction = ml_model.predict(features) * 100  # 抑郁风险（百分比）
        
        # 保存预测结果到数据库
        save_prediction_result(user_id, 'anxiety_risk', anxiety_prediction, ml_model.model_version, features)
        save_prediction_result(user_id, 'depression_risk', depression_prediction, ml_model.model_version, features)
        
        # 生成报告内容
        report = {
            'user_info': {
                'user_id': user.id,
                'username': user.username,
                'class_name': user.class_name,
                'student_id': user.student_id
            },
            'test_results': [
                {
                    'test_type': result.test_type,
                    'score': result.score,
                    'test_date': result.test_date,
                    'interpretation': result.interpretation
                } for result in test_results
            ],
            'emotion_analysis': {
                'recent_records': len(emotion_records),
                'emotion_distribution': analyze_emotion_distribution(emotion_records)
            },
            'ml_predictions': {
                'anxiety_risk': round(anxiety_prediction, 2),
                'depression_risk': round(depression_prediction, 2),
                'model_version': ml_model.model_version,
                'prediction_date': datetime.utcnow().isoformat()
            },
            'personalized_recommendations': generate_recommendations(anxiety_prediction, depression_prediction, emotion_records),
            'report_generated_at': datetime.utcnow().isoformat()
        }
        
        # 记录报告生成事件
        log_system_operation(
            operator_id=user_id,
            operator_type='system',
            operation_name='report_generated',
            operation_details=json.dumps({"user_id": user_id, "report_type": "personalized"}),
            execution_result='success'
        )
        
        return report
    except Exception as e:
        log_system_operation(
            operator_type='system',
            operation_name='report_generation_failure',
            operation_details=json.dumps({"user_id": user_id, "error": str(e)}),
            execution_result='failed'
        )
        raise
    finally:
        session.close()


# 辅助函数：准备预测特征数据
def prepare_prediction_features(user_id):
    """
    为机器学习模型准备预测特征数据
    
    参数:
        user_id: 用户ID
        
    返回:
        特征数据
    """
    session = SessionLocal()
    try:
        # 获取用户的测试分数作为特征
        test_results = session.query(TestResult).filter(TestResult.user_id == user_id).all()
        
        # 获取用户的情感分析记录
        emotion_records = session.query(Conversation).filter(Conversation.user_id == user_id).all()
        
        # 提取特征
        features = []
        
        # 测试分数特征
        sds_scores = [r.score for r in test_results if r.test_type == 'sds']
        sas_scores = [r.score for r in test_results if r.test_type == 'sas']
        
        avg_sds = sum(sds_scores) / len(sds_scores) if sds_scores else 0
        avg_sas = sum(sas_scores) / len(sas_scores) if sas_scores else 0
        
        # 情感分析特征
        negative_emotions = sum(1 for r in emotion_records if r.emotion == 'negative')
        total_emotions = len(emotion_records)
        negative_ratio = negative_emotions / total_emotions if total_emotions else 0
        
        # 特征列表
        features = [
            avg_sds,
            avg_sas,
            negative_ratio,
            len(test_results),
            total_emotions
        ]
        
        return [features]  # 返回二维数组（模型期望的输入格式）
    finally:
        session.close()


# 辅助函数：保存预测结果
def save_prediction_result(user_id, prediction_type, prediction_value, model_version, feature_data):
    """
    保存预测结果到数据库
    
    参数:
        user_id: 用户ID
        prediction_type: 预测类型
        prediction_value: 预测值
        model_version: 模型版本
        feature_data: 特征数据（列表格式）
    """
    session = SessionLocal()
    try:
        prediction = MLPredictionResult(
            user_id=user_id,
            prediction_type=prediction_type,
            prediction_value=prediction_value,
            model_version=model_version,
            feature_data=json.dumps(feature_data)
        )
        session.add(prediction)
        session.commit()
    finally:
        session.close()


# 辅助函数：分析情感分布
def analyze_emotion_distribution(emotion_records):
    """
    分析情感分布
    
    参数:
        emotion_records: 情感记录列表
        
    返回:
        情感分布统计
    """
    distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for record in emotion_records:
        emotion = record.emotion
        if emotion in distribution:
            distribution[emotion] += 1
    
    total = sum(distribution.values())
    if total > 0:
        distribution = {k: round(v / total * 100, 2) for k, v in distribution.items()}
    
    return distribution


# 辅助函数：生成个性化建议
def generate_recommendations(anxiety_prediction, depression_prediction, emotion_records):
    """
    根据分析结果生成个性化建议
    
    参数:
        anxiety_prediction: 焦虑风险预测值
        depression_prediction: 抑郁风险预测值
        emotion_records: 情感记录列表
        
    返回:
        建议列表
    """
    recommendations = []
    
    # 基于焦虑风险的建议
    if anxiety_prediction > 70:
        recommendations.append("您的焦虑风险较高，建议寻求专业心理咨询师的帮助。")
    elif anxiety_prediction > 50:
        recommendations.append("您的焦虑风险中等，建议定期进行放松练习，如深呼吸、冥想等。")
    
    # 基于抑郁风险的建议
    if depression_prediction > 70:
        recommendations.append("您的抑郁风险较高，建议尽快与学校心理咨询中心联系。")
    elif depression_prediction > 50:
        recommendations.append("您的抑郁风险中等，建议保持规律的作息时间，多参加户外活动。")
    
    # 基于情感记录的建议
    negative_emotions = sum(1 for r in emotion_records if r.emotion == 'negative')
    total_emotions = len(emotion_records)
    if total_emotions > 0 and negative_emotions / total_emotions > 0.7:
        recommendations.append("最近您的负面情绪较多，建议尝试记录情绪日记，或与信任的朋友倾诉。")
    
    # 通用建议
    recommendations.append("保持健康的生活方式，包括充足的睡眠、均衡的饮食和适当的运动。")
    
    return recommendations


# 机器学习模型评估与迭代机制
class MLModelEvaluator:
    """
    机器学习模型评估与迭代器
    """
    
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        评估模型性能
        
        参数:
            model: 训练好的机器学习模型
            X_test: 测试特征数据
            y_test: 测试标签数据
            
        返回:
            评估结果字典
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        # 进行预测
        y_pred = model.predict(X_test) > 0.5  # 二分类预测
        y_prob = model.predict(X_test)  # 概率预测
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    @staticmethod
    def cross_validate_model(model, X, y, cv=5):
        """
        交叉验证模型
        
        参数:
            model: 机器学习模型
            X: 特征数据
            y: 标签数据
            cv: 交叉验证折数
            
        返回:
            交叉验证结果字典
        """
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # 定义评估指标
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1_score': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
        }
        
        # 进行交叉验证
        results = cross_validate(model.model, X, y, cv=cv, scoring=scoring, return_train_score=True)
        
        # 计算平均值
        avg_results = {
            f'{metric}_mean': results[metric].mean() for metric in results
        }
        
        return avg_results
    
    @staticmethod
    def retrain_model(user_id=None):
        """
        重新训练模型
        
        参数:
            user_id: 可选，特定用户ID，用于个性化模型
            
        返回:
            训练后的模型和评估结果
        """
        from sklearn.model_selection import train_test_split
        
        try:
            # 准备训练数据
            X, y = MLModelEvaluator.prepare_training_data(user_id)
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # 初始化并训练模型
            ml_model = MentalHealthMLModel(model_type='xgboost')
            ml_model.train(X_train, y_train)
            
            # 评估模型
            metrics = MLModelEvaluator.evaluate_model(ml_model, X_test, y_test)
            
            # 交叉验证
            cv_results = MLModelEvaluator.cross_validate_model(ml_model, X, y)
            
            # 保存模型
            model_path = 'models/mental_health_model.pkl'
            if user_id:
                model_path = f'models/mental_health_model_user_{user_id}.pkl'
            
            ml_model.save_model(model_path)
            
            # 记录模型训练事件
            log_system_operation(
                operator_type='system',
                operation_name='model_retraining',
                operation_details=json.dumps({
                    "model_type": ml_model.model_type,
                    "model_version": ml_model.model_version,
                    "user_id": user_id,
                    "metrics": metrics,
                    "cv_results": cv_results
                }),
                execution_result='success'
            )
            
            return ml_model, metrics, cv_results
            
        except Exception as e:
            log_system_operation(
                operator_type='system',
                operation_name='model_retraining_failure',
                operation_details=json.dumps({"error": str(e), "user_id": user_id}),
                execution_result='failed'
            )
            raise
    
    @staticmethod
    def prepare_training_data(user_id=None):
        """
        准备训练数据
        
        参数:
            user_id: 可选，特定用户ID
            
        返回:
            特征数据和标签数据
        """
        session = SessionLocal()
        try:
            # 查询所有用户的测试结果和情感记录
            query = session.query(TestResult, Conversation)
            query = query.join(User, TestResult.user_id == User.id)
            query = query.join(Conversation, User.id == Conversation.user_id)
            
            if user_id:
                query = query.filter(User.id == user_id)
            
            results = query.all()
            
            X = []
            y = []  # 标签：1表示有心理健康风险，0表示没有
            
            for test_result, conversation in results:
                # 提取特征
                features = []
                
                # 测试分数特征
                features.append(test_result.score)
                
                # 情感特征
                if conversation.emotion == 'negative':
                    features.append(1)
                elif conversation.emotion == 'positive':
                    features.append(3)
                else:
                    features.append(2)
                
                X.append(features)
                
                # 生成标签（示例：分数超过70或负面情绪标记为有风险）
                risk = 1 if test_result.score > 70 or conversation.emotion == 'negative' else 0
                y.append(risk)
            
            import numpy as np
            return np.array(X), np.array(y)
            
        finally:
            session.close()
    
    @staticmethod
    def monitor_model_performance(): 
        """
        监控模型性能，定期检查是否需要重新训练
        
        返回:
            监控结果和是否需要重新训练的建议
        """
        session = SessionLocal()
        try:
            # 获取最近的预测结果和实际结果
            predictions = session.query(MLPredictionResult).order_by(MLPredictionResult.prediction_time.desc()).limit(100).all()
            
            if not predictions:
                return {"message": "没有足够的预测数据进行监控", "needs_retraining": False}
            
            # 计算实际结果（这里需要结合实际的测试结果作为标签）
            y_pred = []
            y_actual = []
            
            for pred in predictions:
                y_pred.append(pred.prediction_value)
                
                # 获取对应的实际测试结果
                test_result = session.query(TestResult)
                test_result = test_result.filter(TestResult.user_id == pred.user_id)
                test_result = test_result.order_by(TestResult.test_date.desc()).first()
                
                if test_result:
                    # 实际风险标签：分数超过70标记为有风险
                    actual_risk = 1 if test_result.score > 70 else 0
                    y_actual.append(actual_risk)
            
            if len(y_actual) < 10:
                return {"message": "实际测试数据不足，无法准确评估模型性能", "needs_retraining": False}
            
            # 计算当前模型性能
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            # 将预测值转换为二分类标签（阈值0.5）
            y_pred_binary = [1 if p > 50 else 0 for p in y_pred]
            
            current_accuracy = accuracy_score(y_actual, y_pred_binary)
            current_roc_auc = roc_auc_score(y_actual, y_pred)
            
            # 定义性能阈值
            ACCURACY_THRESHOLD = 0.7
            ROC_AUC_THRESHOLD = 0.75
            
            # 判断是否需要重新训练
            needs_retraining = current_accuracy < ACCURACY_THRESHOLD or current_roc_auc < ROC_AUC_THRESHOLD
            
            # 记录监控结果
            log_system_operation(
                operator_type='system',
                operation_name='model_performance_monitoring',
                operation_details=json.dumps({
                    "accuracy": current_accuracy,
                    "roc_auc": current_roc_auc,
                    "needs_retraining": needs_retraining,
                    "sample_size": len(y_actual)
                }),
                execution_result='success'
            )
            
            return {
                "current_accuracy": current_accuracy,
                "current_roc_auc": current_roc_auc,
                "needs_retraining": needs_retraining,
                "message": "模型性能监控完成"
            }
            
        except Exception as e:
            log_system_operation(
                operator_type='system',
                operation_name='model_monitoring_failure',
                operation_details=json.dumps({"error": str(e)}),
                execution_result='failed'
            )
            return {
                "error": str(e),
                "needs_retraining": True,  # 监控失败时默认建议重新训练
                "message": "模型性能监控失败"
            }
        finally:
            session.close()


# 模型迭代与更新调度器
def schedule_model_update():
    """
    调度模型定期更新
    """
    import schedule
    import time
    
    def update_model_task():
        """模型更新任务"""
        print("开始执行模型更新任务...")
        try:
            # 检查模型性能
            monitoring_result = MLModelEvaluator.monitor_model_performance()
            
            if monitoring_result["needs_retraining"]:
                print("模型性能下降，开始重新训练...")
                MLModelEvaluator.retrain_model()
                print("模型重新训练完成")
            else:
                print("模型性能良好，无需重新训练")
        except Exception as e:
            print(f"模型更新任务执行失败: {e}")
    
    # 每天凌晨2点执行模型更新任务
    schedule.every().day.at("02:00").do(update_model_task)
    
    # 每周日执行一次完整的交叉验证和重新训练
    schedule.every().sunday.at("03:00").do(lambda: MLModelEvaluator.retrain_model())
    
    print("模型更新调度器已启动")
    
    # 运行调度器
    while True:
        schedule.run_pending()
        time.sleep(60)  # 每分钟检查一次

## 3. AI模块详细设计

### 3.1 模型训练细节

#### 3.1.1 数据集预处理后的特征列表

##### SCL-90 特征提取
| 特征名称 | 描述 | 数据类型 |
|---------|------|---------|
| somatization | 躯体化因子得分 | 数值型 |
|强迫症状 | 强迫症状因子得分 | 数值型 |
|人际关系敏感 | 人际关系敏感因子得分 | 数值型 |
|抑郁 | 抑郁因子得分 | 数值型 |
|焦虑 | 焦虑因子得分 | 数值型 |
|敌对 | 敌对因子得分 | 数值型 |
|恐怖 | 恐怖因子得分 | 数值型 |
|偏执 | 偏执因子得分 | 数值型 |
|精神病性 | 精神病性因子得分 | 数值型 |
|其他 | 其他因子得分 | 数值型 |
|总分 | SCL-90总分 | 数值型 |

##### 综合特征列表
| 特征名称 | 描述 | 数据类型 | 来源 |
|---------|------|---------|------|
| avg_sds | 平均SDS（抑郁自评量表）得分 | 数值型 | 测试结果 |
| avg_sas | 平均SAS（焦虑自评量表）得分 | 数值型 | 测试结果 |
| negative_ratio | 负面情绪占比 | 数值型 | 对话记录 |
| test_count | 测试次数 | 数值型 | 测试结果 |
| emotion_count | 情绪记录次数 | 数值型 | 对话记录 |
| stress_score | 压力源评估得分 | 数值型 | 测试结果 |
| recovery_score | 疫情后恢复得分 | 数值型 | 测试结果 |
| scl_90_somatization | SCL-90躯体化因子 | 数值型 | 测试结果 |
| scl_90_compulsion | SCL-90强迫症状因子 | 数值型 | 测试结果 |
| scl_90_interpersonal | SCL-90人际关系敏感因子 | 数值型 | 测试结果 |
| scl_90_depression | SCL-90抑郁因子 | 数值型 | 测试结果 |
| scl_90_anxiety | SCL-90焦虑因子 | 数值型 | 测试结果 |
| scl_90_hostility | SCL-90敌对因子 | 数值型 | 测试结果 |
| scl_90_phobia | SCL-90恐怖因子 | 数值型 | 测试结果 |
| scl_90_paranoid | SCL-90偏执因子 | 数值型 | 测试结果 |
| scl_90_psychotic | SCL-90精神病性因子 | 数值型 | 测试结果 |
| scl_90_other | SCL-90其他因子 | 数值型 | 测试结果 |
| scl_90_total | SCL-90总分 | 数值型 | 测试结果 |

#### 3.1.2 模型评估指标

| 指标名称 | 描述 | 目标值 |
|---------|------|---------|
| accuracy | 准确率 | ≥80% |
| precision | 精确率 | ≥75% |
| recall | 召回率 | ≥75% |
| f1_score | F1分数 | ≥75% |
| roc_auc | ROC曲线下面积 | ≥0.85 |

#### 3.1.3 实验过程

1. **数据集划分**
   - 训练集：70%（用于模型训练）
   - 验证集：15%（用于超参数调优）
   - 测试集：15%（用于最终模型评估）
   - 划分方式：分层抽样，保持正负样本比例一致

2. **特征工程**
   - 缺失值处理：使用均值/中位数填充
   - 异常值处理：使用IQR方法检测并修正异常值
   - 特征缩放：使用StandardScaler进行标准化
   - 特征选择：使用随机森林特征重要性和L1正则化进行特征选择

3. **超参数调优**
   - 调优方法：网格搜索（Grid Search）+ 5折交叉验证
   - 搜索空间：
     - 随机森林：n_estimators（100-500），max_depth（5-20），min_samples_split（2-10）
     - XGBoost：n_estimators（100-500），learning_rate（0.01-0.1），max_depth（3-10），subsample（0.6-1.0）
   - 调优目标：最大化F1分数

4. **模型选择**
   - 基线模型：逻辑回归
   - 候选模型：随机森林、XGBoost、LightGBM
   - 最终模型：XGBoost（基于交叉验证结果选择）

5. **模型训练与验证**
   - 使用训练集进行模型训练
   - 使用验证集进行超参数调优
   - 使用测试集进行最终模型评估
   - 保存最佳模型到文件系统

### 3.2 预测逻辑触发时机

#### 3.2.1 实时预测
1. **用户完成测试后**
   - 触发条件：用户提交心理测试
   - 触发流程：
     - 保存测试结果到数据库
     - 调用机器学习模型进行预测
     - 保存预测结果到数据库
     - 返回预测结果给前端
   - 预测类型：焦虑风险、抑郁风险

2. **情感分析后**
   - 触发条件：用户与系统完成对话
   - 触发流程：
     - 保存对话记录到数据库
     - 进行情感分析
     - 根据情感分析结果决定是否需要预测
     - 如需预测，调用机器学习模型
     - 保存预测结果到数据库
   - 预测类型：情绪趋势预测

#### 3.2.2 批量预测
1. **每日批量预测**
   - 触发时间：每天凌晨2点
   - 触发流程：
     - 检查模型性能
     - 如需重新训练，先训练模型
     - 对所有用户进行预测
     - 更新预测结果到数据库
   - 预测类型：综合心理健康风险

2. **每周深度预测**
   - 触发时间：每周日凌晨3点
   - 触发流程：
     - 执行完整的模型训练和交叉验证
     - 对所有用户进行深度预测
     - 生成周度预测报告
     - 保存预测结果到数据库
   - 预测类型：长期心理健康趋势

#### 3.2.3 手动触发预测
1. **管理员手动触发**
   - 触发方式：管理员在系统配置页面点击"立即预测"按钮
   - 触发流程：
     - 验证管理员权限
     - 调用机器学习模型
     - 对指定用户或所有用户进行预测
     - 保存预测结果到数据库
     - 返回预测结果给管理员
   - 预测类型：按需选择

### 3.3 模型部署与更新机制

#### 3.3.1 模型部署
1. **模型文件格式**：使用Pickle格式保存训练好的模型
2. **部署位置**：系统服务器的models目录下
3. **版本管理**：每个模型文件包含版本号，如"mental_health_model_v1.0.pkl"
4. **加载方式**：系统启动时加载模型到内存

#### 3.3.2 模型更新
1. **自动更新**：通过调度器定期检查模型性能，如需更新则自动重新训练
2. **手动更新**：管理员可在系统配置页面手动触发模型更新
3. **更新流程**：
   - 训练新模型
   - 评估新模型性能
   - 如性能优于当前模型，则替换当前模型
   - 保存模型更新日志
4. **回滚机制**：保存最近3个版本的模型，支持回滚到之前的版本

### 3.4 模型解释性设计

#### 3.4.1 特征重要性分析
1. **实现方式**：使用SHAP（SHapley Additive exPlanations）值计算特征重要性
2. **展示方式**：
   - 条形图展示全局特征重要性
   - 热力图展示特征间的交互作用
   - 瀑布图展示单个预测的特征贡献

#### 3.4.2 预测结果解释
1. **实现方式**：为每个预测结果生成自然语言解释
2. **解释内容**：
   - 影响预测结果的主要特征
   - 各特征的贡献程度
   - 预测结果的可信度
   - 基于预测结果的建议

#### 3.4.3 模型透明度报告
1. **生成频率**：每月生成一次
2. **报告内容**：
   - 模型性能指标
   - 特征重要性分布
   - 预测结果分布
   - 模型偏差分析
   - 模型更新日志

## 4. 前端可视化具体方案

### 4.1 ECharts图表类型及应用场景

#### 4.1.1 学生个人报告

##### 1. 雷达图 - SCL-90各因子得分
- **图表类型**：雷达图
- **应用场景**：直观展示学生在SCL-90的10个因子上的得分分布
- **数据来源**：测试结果表中的SCL-90因子得分
- **展示内容**：
  - 10个因子的得分（躯体化、强迫症状、人际关系敏感、抑郁、焦虑、敌对、恐怖、偏执、精神病性、其他）
  - 正常参考范围（以虚线表示）
  - 学生实际得分（以实线表示）
- **设计要点**：
  - 每个因子使用不同颜色区分
  - 得分越高，雷达图面积越大
  - 超出正常范围的因子用醒目的颜色标记

##### 2. 折线图 - 历史测试趋势
- **图表类型**：折线图
- **应用场景**：展示学生在一段时间内的测试得分变化趋势
- **数据来源**：测试结果表中的历史得分数据
- **展示内容**：
  - X轴：测试日期
  - Y轴：测试得分
  - 多条折线：分别展示不同测试类型（SDS、SAS、SCL-90等）的得分趋势
- **设计要点**：
  - 每条折线使用不同颜色区分
  - 显示数据点的具体数值
  - 支持按测试类型筛选
  - 显示趋势线，预测未来得分变化

##### 3. 柱状图 - 单项测试得分对比
- **图表类型**：柱状图
- **应用场景**：对比学生在单次测试中各维度的得分
- **数据来源**：测试结果表中的单项测试数据
- **展示内容**：
  - X轴：测试维度（如压力源评估的5个维度）
  - Y轴：得分
  - 柱状图：展示每个维度的具体得分
  - 平均值参考线：展示同年级学生的平均得分
- **设计要点**：
  - 超出平均值的柱子用绿色表示
  - 低于平均值的柱子用红色表示
  - 显示每个柱子的具体数值

##### 4. 饼图 - 情绪分布
- **图表类型**：饼图
- **应用场景**：展示学生在对话记录中的情绪分布
- **数据来源**：对话记录表中的情感分析结果
- **展示内容**：
  - 饼图扇区：积极情绪、消极情绪、中性情绪
  - 每个扇区的占比百分比
- **设计要点**：
  - 使用柔和的颜色区分不同情绪
  - 扇区大小与情绪占比成正比
  - 显示每个扇区的具体数值和百分比

#### 4.1.2 管理员统计页

##### 1. 热力图 - 班级心理状态分布
- **图表类型**：热力图
- **应用场景**：展示不同班级的心理状态分布情况
- **数据来源**：测试结果表中的班级测试数据
- **展示内容**：
  - X轴：班级名称
  - Y轴：时间（周/月）
  - 颜色深浅：表示该班级在该时间段的心理状态（颜色越深，状态越差）
- **设计要点**：
  - 使用从绿色到红色的渐变颜色
  - 支持按年级筛选
  - 悬停时显示详细信息（班级、时间、平均得分、异常人数）

##### 2. 柱状图 - 不同年级压力源对比
- **图表类型**：分组柱状图
- **应用场景**：对比不同年级学生在各压力源上的得分
- **数据来源**：测试结果表中的压力源评估数据
- **展示内容**：
  - X轴：压力源维度（学业压力、社交压力、生活压力、疫情影响、未来规划）
  - Y轴：平均得分
  - 分组：不同年级（如2020级、2021级、2022级）
- **设计要点**：
  - 每个年级使用不同颜色区分
  - 显示每个柱子的具体数值
  - 支持按压力源维度筛选

##### 3. 散点图 - 心理健康风险分布
- **图表类型**：散点图
- **应用场景**：展示所有学生的心理健康风险分布
- **数据来源**：机器学习预测结果表中的预测数据
- **展示内容**：
  - X轴：焦虑风险得分
  - Y轴：抑郁风险得分
  - 散点大小：表示风险等级
  - 颜色：表示年级或班级
- **设计要点**：
  - 添加风险等级划分线（如高风险、中风险、低风险）
  - 支持按年级、班级筛选
  - 悬停时显示学生基本信息和具体风险得分

##### 4. 堆叠柱状图 - 测试类型分布
- **图表类型**：堆叠柱状图
- **应用场景**：展示不同测试类型的完成情况
- **数据来源**：测试结果表中的测试类型数据
- **展示内容**：
  - X轴：时间（月/季度）
  - Y轴：测试完成数量
  - 堆叠：不同测试类型的完成数量
- **设计要点**：
  - 每个测试类型使用不同颜色区分
  - 显示每个堆叠部分的具体数值
  - 支持按测试类型筛选

### 4.2 交互设计

#### 4.2.1 图表筛选条件
1. **时间筛选**：
   - 提供日期选择器，支持按日、周、月、季度、年筛选
   - 预设常用时间范围（如最近7天、最近30天、本学期）
   - 支持自定义时间范围

2. **测试类型筛选**：
   - 提供多选框，支持选择一种或多种测试类型
   - 预设常用筛选组合（如所有测试、仅情绪测试）

3. **年级/班级筛选**：
   - 提供级联选择器，先选择年级，再选择班级
   - 支持全选/取消全选功能

4. **风险等级筛选**：
   - 提供滑块或单选按钮，支持按风险等级筛选
   - 风险等级：低风险、中风险、高风险

#### 4.2.2 悬停提示信息
1. **基础信息**：
   - 显示数据点的具体数值
   - 显示数据的时间信息
   - 显示相关的描述性文字

2. **详细信息**：
   - 学生基本信息（姓名、班级、学号）
   - 测试的具体内容和解释
   - 基于该数据的建议

3. **交互效果**：
   - 悬停时显示提示框
   - 提示框使用半透明背景，不遮挡主要数据
   - 提示框内容简洁明了，重点突出

#### 4.2.3 下钻功能
1. **班级 → 学生**：
   - 点击班级热力图中的某一格，可下钻到该班级的学生列表
   - 显示该班级所有学生的心理健康状态
   - 支持按学生姓名搜索

2. **年级 → 班级 → 学生**：
   - 在年级统计图表中，点击某个年级，可下钻到该年级的班级列表
   - 点击某个班级，可下钻到该班级的学生列表
   - 点击某个学生，可查看该学生的详细报告

3. **测试类型 → 具体测试**：
   - 在测试类型分布图表中，点击某个测试类型，可下钻到该测试类型的具体测试列表
   - 显示该测试类型的所有测试记录
   - 支持按得分排序

#### 4.2.4 导出功能
1. **图表导出**：
   - 支持将图表导出为PNG、JPG、SVG等格式
   - 支持自定义导出图片的尺寸和分辨率
   - 导出的图片包含图表标题和图例

2. **数据导出**：
   - 支持将图表数据导出为CSV、Excel等格式
   - 导出的数据包含原始数据和统计数据
   - 支持选择导出的数据范围和字段

3. **报告导出**：
   - 支持将完整报告导出为PDF格式
   - PDF报告包含所有图表和文字说明
   - 支持自定义报告模板

### 4.3 响应式设计

#### 4.3.1 适配不同设备
1. **桌面端**：
   - 完整展示所有图表和数据
   - 图表尺寸较大，细节清晰
   - 支持多图表并排展示

2. **平板端**：
   - 调整图表布局，改为上下排列
   - 简化部分交互功能
   - 保持图表的可读性

3. **移动端**：
   - 仅展示核心图表
   - 简化图表细节，突出重点数据
   - 支持横向滚动查看完整图表
   - 优化触摸交互

#### 4.3.2 适配不同屏幕尺寸
1. **自动调整**：
   - 图表尺寸随容器大小自动调整
   - 字体大小随屏幕尺寸自动缩放
   - 图例位置自动调整

2. **断点设计**：
   - 针对不同屏幕尺寸设置断点
   - 在每个断点处调整图表布局和样式
   - 确保在任何屏幕尺寸下都有良好的显示效果

### 4.4 性能优化

#### 4.4.1 数据加载优化
1. **分页加载**：
   - 大量数据采用分页加载
   - 每次只加载当前视图所需的数据
   - 滚动到页面底部时自动加载下一页

2. **懒加载**：
   - 图表在可见区域内才开始渲染
   - 减少初始加载时间
   - 提高页面响应速度

3. **缓存机制**：
   - 缓存已加载的数据
   - 避免重复请求相同数据
   - 支持手动刷新数据

#### 4.4.2 渲染优化
1. **减少图表数量**：
   - 在同一页面上避免过多图表
   - 采用标签页或折叠面板组织图表
   - 只显示用户当前需要查看的图表

2. **简化图表样式**：
   - 减少不必要的动画效果
   - 简化图表的颜色和样式
   - 优化图表的渲染性能

3. **异步渲染**：
   - 图表采用异步渲染方式
   - 不阻塞页面的其他内容加载
   - 提高页面的整体加载速度

## 5. 测试计划与验收标准

### 5.1 功能测试

#### 5.1.1 用户认证与权限管理

| 测试用例 | 测试步骤 | 预期结果 |
|---------|---------|---------|
| 正确密码登录 | 1. 输入正确的用户名和密码<br>2. 点击"登录"按钮 | 登录成功，跳转到对应角色首页 |
| 错误密码登录 | 1. 输入正确的用户名和错误的密码<br>2. 点击"登录"按钮 | 登录失败，显示错误提示"用户名或密码错误" |
| 空用户名登录 | 1. 不输入用户名，只输入密码<br>2. 点击"登录"按钮 | 前端提示"请输入用户名"，不发送请求 |
| 空密码登录 | 1. 输入用户名，不输入密码<br>2. 点击"登录"按钮 | 前端提示"请输入密码"，不发送请求 |
| 权限不足访问 | 1. 学生用户尝试访问管理员页面<br>2. 直接输入管理员页面URL | 跳转到403页面，显示"权限不足"提示 |
| 注销功能 | 1. 登录成功后点击"注销"按钮 | 注销成功，清除会话，跳转到登录页面 |

#### 5.1.2 心理测试功能

| 测试用例 | 测试步骤 | 预期结果 |
|---------|---------|---------|
| 测试列表展示 | 1. 登录学生账号<br>2. 进入测试列表页面 | 正确展示所有可用的测试类型 |
| 测试说明展示 | 1. 从测试列表选择一个测试类型<br>2. 进入测试说明页面 | 正确展示测试说明、注意事项和测试协议 |
| 答题页面展示 | 1. 点击"开始测试"按钮<br>2. 进入答题页面 | 正确展示题目，支持上一题/下一题导航 |
| 答案实时保存 | 1. 回答几道题目<br>2. 刷新页面 | 之前的答案被正确保存，可继续作答 |
| 测试提交成功 | 1. 完成所有题目<br>2. 点击"提交"按钮 | 测试提交成功，跳转到结果页面 |
| 未完成题目提示 | 1. 回答部分题目<br>2. 点击"提交"按钮 | 提示"还有未完成的题目，请继续作答" |
| 测试结果生成 | 1. 测试提交成功后<br>2. 查看结果页面 | 正确显示测试得分、结果解释和个性化建议 |

#### 5.1.3 报告生成功能

| 测试用例 | 测试步骤 | 预期结果 |
|---------|---------|---------|
| 报告生成成功 | 1. 从结果页面点击"查看报告"<br>2. 进入报告生成页面 | 成功生成个性化报告，包含测试结果和预测数据 |
| 报告可视化展示 | 1. 查看生成的报告<br>2. 检查图表展示 | 正确展示雷达图、折线图等可视化图表 |
| 报告PDF导出 | 1. 在报告页面点击"导出PDF"<br>2. 选择保存路径 | 成功导出PDF格式报告，包含所有内容 |
| 报告数据准确性 | 1. 对比报告数据与测试结果<br>2. 检查预测结果 | 报告数据与测试结果一致，预测结果合理 |

#### 5.1.4 管理员功能

| 测试用例 | 测试步骤 | 预期结果 |
|---------|---------|---------|
| 用户列表展示 | 1. 登录管理员账号<br>2. 进入用户管理页面 | 正确展示所有用户，支持分页、筛选和搜索 |
| 用户创建成功 | 1. 点击"创建用户"按钮<br>2. 填写用户信息<br>3. 点击"保存" | 成功创建用户，显示在用户列表中 |
| 用户编辑成功 | 1. 选择一个用户<br>2. 点击"编辑"按钮<br>3. 修改用户信息<br>4. 点击"保存" | 成功修改用户信息，更新用户列表 |
| 用户禁用成功 | 1. 选择一个用户<br>2. 点击"禁用"按钮<br>3. 确认操作 | 成功禁用用户，用户状态变为"inactive" |
| 统计数据展示 | 1. 进入统计分析页面<br>2. 查看统计图表 | 正确展示各种统计图表，数据准确 |
| 系统日志查看 | 1. 进入系统日志页面<br>2. 查看日志记录 | 正确展示系统操作日志，支持筛选和搜索 |

### 5.2 性能测试

#### 5.2.1 页面加载性能

| 测试项 | 测试条件 | 预期结果 |
|-------|---------|---------|
| 登录页面加载时间 | 正常网络环境 | ≤1秒 |
| 首页加载时间 | 正常网络环境 | ≤2秒 |
| 测试列表页面加载时间 | 正常网络环境 | ≤2秒 |
| 答题页面加载时间 | 正常网络环境 | ≤2秒 |
| 结果页面加载时间 | 正常网络环境 | ≤3秒 |
| 报告页面加载时间 | 正常网络环境 | ≤3秒 |
| 管理员统计页面加载时间 | 正常网络环境，1000条数据 | ≤3秒 |

#### 5.2.2 数据处理性能

| 测试项 | 测试条件 | 预期结果 |
|-------|---------|---------|
| 测试提交处理时间 | 单用户提交20题测试 | ≤1秒 |
| 报告生成处理时间 | 单用户生成报告 | ≤2秒 |
| 批量预测处理时间 | 1000个用户批量预测 | ≤30秒 |
| 数据导入处理时间 | 1000条学生记录导入 | ≤10秒 |
| 数据查询处理时间 | 查询单个用户50条测试记录 | ≤1秒 |

#### 5.2.3 并发性能

| 测试项 | 测试条件 | 预期结果 |
|-------|---------|---------|
| 并发登录 | 50个用户同时登录 | 登录成功率≥99%，平均响应时间≤2秒 |
| 并发测试 | 20个用户同时进行测试 | 系统稳定运行，无崩溃，平均响应时间≤3秒 |
| 并发报告生成 | 10个用户同时生成报告 | 报告生成成功率≥99%，平均响应时间≤3秒 |

### 5.3 验收标准

#### 5.3.1 功能验收标准

| 验收项 | 验收标准 |
|-------|---------|
| 系统部署 | 成功部署系统，可通过浏览器访问 |
| 用户认证 | 实现学生和管理员的登录、注销功能，权限控制正确 |
| 心理测试 | 实现至少3类心理测试（SDS、SAS、SCL-90），支持答题、提交、结果展示 |
| 报告生成 | 实现个性化报告生成功能，包含测试结果和预测数据 |
| 数据管理 | 实现用户管理、测试数据管理功能 |
| 统计分析 | 实现至少4种统计图表，支持筛选和导出 |
| 机器学习 | 成功部署至少1个机器学习模型，支持实时预测和批量预测 |
| 数据可视化 | 成功实现ECharts可视化，包括雷达图、折线图、柱状图等 |

#### 5.3.2 性能验收标准

| 验收项 | 验收标准 |
|-------|---------|
| 页面加载 | 所有页面加载时间≤3秒 |
| 数据处理 | 测试提交、报告生成等操作响应时间≤3秒 |
| 并发处理 | 50并发用户下系统稳定运行，成功率≥99% |
| 数据规模 | 支持至少1000条学生记录、5000条测试记录的存储和查询 |

#### 5.3.3 代码质量验收标准

| 验收项 | 验收标准 |
|-------|---------|
| 代码规范 | 代码符合PEP 8规范，无明显语法错误 |
| 注释完善 | 关键代码有详细注释，文档齐全 |
| 模块化设计 | 代码结构清晰，模块化程度高，易于维护 |
| 异常处理 | 实现了完善的异常处理机制 |
| 安全性 | 实现了必要的安全措施，如密码加密、SQL注入防护等 |

#### 5.3.4 文档验收标准

| 验收项 | 验收标准 |
|-------|---------|
| 需求文档 | 完整的需求分析，包含功能需求、非功能需求等 |
| 设计文档 | 完整的系统设计，包含架构设计、数据库设计、AI模块设计等 |
| 测试文档 | 完整的测试计划和测试用例，包含功能测试、性能测试等 |
| 用户手册 | 完整的用户操作手册，包含系统功能说明和操作步骤 |
| 部署文档 | 完整的系统部署文档，包含环境搭建、配置说明等 |

## 6. 项目进度与分工

### 6.1 开发阶段里程碑

由于本项目为个人毕设项目，采用开发阶段里程碑的方式进行进度管理，具体如下：

| 阶段 | 时间规划 | 主要任务 | 交付物 |
|------|---------|---------|--------|
| **需求分析与设计** | 第1-2周 | 1. 确定系统目标和功能需求<br>2. 进行系统架构设计<br>3. 完成数据库设计<br>4. 制定详细开发计划 | 需求文档、系统设计文档、数据库设计文档 |
| **环境搭建与基础开发** | 第3周 | 1. 搭建开发环境<br>2. 实现用户认证与权限管理<br>3. 创建数据库表结构<br>4. 实现基础API框架 | 可运行的基础系统、数据库表结构、API文档 |
| **心理测试功能开发** | 第4-5周 | 1. 实现测试列表展示<br>2. 开发答题页面<br>3. 实现测试提交功能<br>4. 开发结果页面 | 完整的心理测试功能模块 |
| **AI模块开发** | 第6-7周 | 1. 数据预处理和特征工程<br>2. 模型训练和评估<br>3. 实现预测API<br>4. 集成到系统中 | 训练好的机器学习模型、预测API、模型评估报告 |
| **报告生成与可视化开发** | 第8-9周 | 1. 实现报告生成功能<br>2. 开发ECharts可视化图表<br>3. 实现报告导出功能<br>4. 优化前端界面 | 完整的报告生成功能、可视化图表、报告导出功能 |
| **管理员功能开发** | 第10周 | 1. 实现用户管理功能<br>2. 开发统计分析页面<br>3. 实现系统日志功能<br>4. 开发系统配置页面 | 完整的管理员功能模块 |
| **测试与优化** | 第11周 | 1. 功能测试<br>2. 性能测试<br>3. 修复bug<br>4. 优化系统性能 | 测试报告、bug修复记录、性能优化报告 |
| **文档编写与部署** | 第12周 | 1. 编写用户手册<br>2. 编写部署文档<br>3. 系统部署<br>4. 最终测试 | 用户手册、部署文档、可运行的系统 |
| **毕设答辩准备** | 第13周 | 1. 准备答辩PPT<br>2. 系统演示准备<br>3. 答辩演练<br>4. 最终修改 | 答辩PPT、系统演示脚本 |

### 6.2 技术栈学习计划

为确保项目顺利进行，制定以下技术栈学习计划：

| 技术 | 学习时间 | 学习内容 | 学习方式 |
|------|---------|---------|----------|
| Flask | 第1周 | 1. Flask基础<br>2. RESTful API开发<br>3. SQLAlchemy ORM | 文档学习、实践项目 |
| ECharts | 第2周 | 1. ECharts基础<br>2. 常用图表类型<br>3. 交互功能开发 | 文档学习、示例代码 |
| 机器学习 | 第3-4周 | 1. scikit-learn基础<br>2. XGBoost算法<br>3. 模型评估方法 | 课程学习、实践项目 |
| Bootstrap | 第2周 | 1. Bootstrap基础<br>2. 响应式设计<br>3. 组件使用 | 文档学习、实践项目 |
| SQLite/MySQL | 第1周 | 1. 数据库基础<br>2. SQL语句<br>3. 索引优化 | 文档学习、实践操作 |

### 6.3 进度管理措施

1. **每日进度记录**：每天记录开发进度和遇到的问题
2. **每周总结**：每周日进行本周工作总结和下周计划调整
3. **风险预警**：及时识别和处理项目风险，调整开发计划
4. **版本控制**：使用Git进行代码版本控制，定期提交代码
5. **测试驱动开发**：先编写测试用例，再进行功能开发
6. **持续集成**：实现代码自动测试和构建

### 6.4 质量保证措施

1. **代码规范**：遵循PEP 8代码规范，使用flake8进行代码检查
2. **单元测试**：对核心功能编写单元测试，覆盖率≥80%
3. **集成测试**：对系统进行集成测试，确保模块间正常协作
4. **用户测试**：邀请同学进行用户测试，收集反馈
5. **代码评审**：邀请导师或同学进行代码评审，提高代码质量
6. **安全测试**：对系统进行安全测试，防止常见安全漏洞

## 7. 风险与应对措施

### 7.1 数据相关风险

| 风险 | 可能性 | 影响程度 | 应对措施 |
|------|--------|---------|----------|
| Kaggle数据集下载失败 | 中 | 高 | 1. 提前下载并备份多个数据集<br>2. 准备本地备用模拟数据<br>3. 实现数据导入的容错机制<br>4. 考虑使用其他公开数据集作为替代 |
| 数据集质量不佳 | 中 | 中 | 1. 进行严格的数据清洗和预处理<br>2. 去除噪声和异常值<br>3. 进行数据质量评估<br>4. 考虑数据增强技术 |
| 数据量不足 | 低 | 中 | 1. 进行数据合成和增强<br>2. 考虑使用迁移学习<br>3. 降低模型复杂度<br>4. 重点关注模型的泛化能力 |
| 数据隐私问题 | 低 | 高 | 1. 严格遵守数据使用协议<br>2. 对敏感数据进行脱敏处理<br>3. 实现数据访问控制<br>4. 仅使用匿名化数据进行模型训练 |

### 7.2 技术开发风险

| 风险 | 可能性 | 影响程度 | 应对措施 |
|------|--------|---------|----------|
| 机器学习模型效果不佳 | 中 | 高 | 1. 先实现简单的基线模型（如逻辑回归）<br>2. 进行充分的特征工程<br>3. 尝试多种模型算法<br>4. 进行超参数调优<br>5. 考虑模型融合技术 |
| 前端可视化难度超预期 | 中 | 中 | 1. 优先实现核心图表功能<br>2. 参考ECharts官方示例<br>3. 简化复杂交互<br>4. 分阶段实现可视化功能<br>5. 考虑使用现成的可视化模板 |
| 系统性能问题 | 低 | 中 | 1. 进行数据库索引优化<br>2. 实现缓存机制<br>3. 优化API响应时间<br>4. 考虑使用异步处理<br>5. 进行性能测试和瓶颈分析 |
| 技术栈学习曲线陡峭 | 低 | 中 | 1. 提前学习核心技术<br>2. 参考成熟的开源项目<br>3. 分阶段学习，边学边用<br>4. 遇到问题及时查阅文档和社区 |

### 7.3 项目管理风险

| 风险 | 可能性 | 影响程度 | 应对措施 |
|------|--------|---------|----------|
| 开发进度延误 | 中 | 高 | 1. 制定详细的开发计划和里程碑<br>2. 每日记录进度，及时调整<br>3. 优先实现核心功能<br>4. 避免范围蔓延<br>5. 预留缓冲时间 |
| 需求变更 | 低 | 中 | 1. 前期充分分析需求<br>2. 与导师保持良好沟通<br>3. 采用敏捷开发方式，灵活应对变更<br>4. 评估变更对进度的影响，合理调整计划 |
| 代码质量下降 | 低 | 中 | 1. 遵循代码规范<br>2. 定期进行代码评审<br>3. 编写单元测试<br>4. 使用代码质量工具进行检查<br>5. 保持代码模块化和可读性 |
| 系统部署问题 | 低 | 中 | 1. 提前测试部署流程<br>2. 编写详细的部署文档<br>3. 考虑使用容器化部署（如Docker）<br>4. 准备备用部署方案 |

### 7.4 其他风险

| 风险 | 可能性 | 影响程度 | 应对措施 |
|------|--------|---------|----------|
| 硬件资源不足 | 低 | 低 | 1. 优先使用云服务进行模型训练<br>2. 优化模型训练代码<br>3. 考虑使用轻量级模型<br>4. 利用学校提供的计算资源 |
| 答辩准备不充分 | 低 | 高 | 1. 提前准备答辩材料<br>2. 进行多次答辩演练<br>3. 熟悉系统的每一个功能<br>4. 准备常见问题的解答<br>5. 与导师和同学进行模拟答辩 |
| 意外情况（如电脑故障） | 低 | 高 | 1. 定期备份代码和数据<br>2. 使用版本控制系统（Git）<br>3. 准备备用电脑<br>4. 云端存储重要文件 |

## 8. 参考文献与技术选型说明

### 8.1 技术选型说明

#### 8.1.1 后端框架：Flask 2.3

**选型理由：**
1. **轻量级设计**：Flask相比Django更轻量级，核心功能简洁，适合小型项目和快速开发
2. **灵活扩展**：通过扩展包可以灵活添加所需功能，如SQLAlchemy、Flask-JWT-Extended等
3. **易于学习**：API设计简洁明了，学习曲线平缓，适合个人毕设项目
4. **良好的社区支持**：拥有丰富的文档和活跃的社区，遇到问题容易找到解决方案
5. **适合RESTful API开发**：内置支持RESTful API开发，适合前后端分离架构
6. **低资源消耗**：运行时资源消耗低，适合在开发环境和演示环境中使用

**对比Django：**
- Django是全功能框架，内置ORM、Admin后台、表单处理等功能，适合大型项目
- Flask是微框架，只提供核心功能，其他功能通过扩展实现，更加灵活
- 对于毕设项目来说，Flask的轻量级和灵活性更适合快速开发和迭代

#### 8.1.2 ORM框架：SQLAlchemy 2.0

**选型理由：**
1. **功能强大**：支持多种数据库后端，提供完整的ORM功能
2. **灵活的查询API**：支持原生SQL查询和ORM查询，满足不同场景需求
3. **良好的文档**：拥有详细的官方文档和丰富的示例
4. **活跃的开发**：持续更新和维护，支持最新的Python版本
5. **与Flask良好集成**：通过Flask-SQLAlchemy扩展可以方便地与Flask集成

#### 8.1.3 数据库：SQLite（开发环境）/ MySQL（可选）

**选型理由（SQLite）：**
1. **无需额外服务器**：SQLite是文件型数据库，不需要安装和配置数据库服务器
2. **易于使用**：开箱即用，适合开发和演示环境
3. **轻量级**：数据库文件体积小，易于备份和迁移
4. **良好的Python支持**：Python内置SQLite支持，无需额外安装驱动
5. **适合小型数据量**：对于毕设演示所需的1000条学生记录和5000条测试记录，性能足够

**选型理由（MySQL，可选）：**
1. **适合生产环境**：MySQL是成熟的关系型数据库，适合生产环境使用
2. **更好的并发性能**：相比SQLite，MySQL支持更好的并发访问
3. **更丰富的功能**：支持存储过程、触发器、视图等高级功能
4. **良好的扩展性**：可以通过主从复制、分库分表等方式扩展

#### 8.1.4 前端技术栈

**Bootstrap 5**
- **响应式设计**：内置响应式网格系统，适合不同设备
- **丰富的组件**：提供大量UI组件，如按钮、表单、导航等
- **易于定制**：支持自定义主题和样式
- **良好的浏览器兼容性**：支持主流浏览器

**ECharts 5**
- **丰富的图表类型**：支持折线图、柱状图、雷达图、热力图等多种图表
- **良好的交互体验**：支持拖拽、缩放、悬停提示等交互功能
- **易于集成**：提供简单的API，易于与前端框架集成
- **良好的文档和示例**：拥有详细的官方文档和丰富的示例

**Jinja2模板**
- **与Flask无缝集成**：Flask默认使用Jinja2模板引擎
- **强大的模板功能**：支持模板继承、宏、过滤器等功能
- **易于学习**：语法简洁明了，易于上手

**原生JavaScript**
- **无需额外依赖**：浏览器内置支持，无需加载额外库
- **更好的性能**：相比框架，原生JavaScript性能更好
- **适合小型项目**：对于毕设项目来说，原生JavaScript足够满足需求

#### 8.1.5 AI模块技术栈

**scikit-learn**
- **易于使用**：提供简单一致的API，适合快速开发
- **丰富的算法**：支持分类、回归、聚类等多种机器学习算法
- **良好的文档**：拥有详细的官方文档和丰富的示例
- **活跃的社区**：拥有活跃的社区和大量的第三方资源

**XGBoost**
- **高性能**：采用梯度提升算法，性能优于传统机器学习算法
- **良好的扩展性**：支持分布式训练，适合大规模数据
- **广泛应用**：在各种机器学习竞赛中取得良好成绩
- **支持Python接口**：提供Python接口，易于与其他Python库集成

**Hugging Face Transformers（可选）**
- **预训练模型丰富**：提供大量预训练模型，如BERT、GPT等
- **易于使用**：提供简单的API，易于加载和使用预训练模型
- **支持多种任务**：支持文本分类、情感分析、问答等多种NLP任务

### 8.2 参考文献

#### 技术文档
1. Flask官方文档：https://flask.palletsprojects.com/
2. SQLAlchemy官方文档：https://docs.sqlalchemy.org/
3. Bootstrap官方文档：https://getbootstrap.com/
4. ECharts官方文档：https://echarts.apache.org/
5. scikit-learn官方文档：https://scikit-learn.org/
6. XGBoost官方文档：https://xgboost.readthedocs.io/
7. Hugging Face Transformers文档：https://huggingface.co/docs/transformers/
8. SQLite官方文档：https://www.sqlite.org/docs.html

#### 教程与书籍
1. 《Flask Web开发实战》
2. 《Python机器学习基础教程》
3. 《ECharts数据可视化》
4. 《SQLAlchemy从入门到精通》

#### 数据集来源
1. Student Mental Health Dataset：https://www.kaggle.com/datasets/ketangangal/studentmentalhealth
2. Anxiety and Depression Dataset：https://www.kaggle.com/datasets/anjaliagrawal1112000/anxiety-and-depression-dataset
3. Stress Level Detection Dataset：https://www.kaggle.com/datasets/kreeshrajani/stress-level-detection-dataset
4. College Student Mental Health：https://www.kaggle.com/datasets/rxnach/college-student-mental-health

#### 学术论文（参考）
1. 大学生心理健康状况调查与分析
2. 基于机器学习的心理健康预测模型研究
3. 心理测试数据的可视化分析方法
4. 基于Python的Web应用开发技术研究


# 数据访问装饰器，用于自动记录数据访问日志
def log_data_access_decorator(action_type, resource_type):
    """
    数据访问装饰器，自动记录数据访问日志
    
    参数:
        action_type: 查询类型（user_query, admin_query, system_query）
        resource_type: 资源类型（conversation, test_result, user_info, system_setting）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 提取用户ID（假设kwargs中包含user_id）
            user_id = kwargs.get('user_id')
            if not user_id and args:
                # 尝试从参数中获取用户ID
                user_id = args[0] if args else None
            
            # 提取资源ID
            resource_id = kwargs.get('resource_id')
            
            # 记录日志前的准备
            start_time = time.time()
            
            try:
                # 执行原始函数
                result = func(*args, **kwargs)
                
                # 记录成功日志
                log_data_access(
                    user_id=user_id,
                    action_type=action_type,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    access_result='success'
                )
                
                return result
            except Exception as e:
                # 记录失败日志
                log_data_access(
                    user_id=user_id,
                    action_type=action_type,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    access_result='failed',
                    error_message=str(e)
                )
                raise
        return wrapper
    return decorator


# 测试进度保存与冲突处理函数
def save_test_progress(user_id, test_type, current_question, answers, device_info=None):
    """
    保存测试进度，并处理多设备同时答题的冲突
    
    参数:
        user_id: 用户ID
        test_type: 测试类型
        current_question: 当前题目编号
        answers: 已答题目（JSON格式）
        device_info: 设备标识
    
    返回:
        tuple: (是否保存成功, 消息)
    
    同步逻辑与冲突解决优先级:
    1. 前端优先使用localStorage本地保存，确保即时响应
    2. 每答完3题或30秒自动同步到后端test_session表
    3. 冲突解决策略：
       - 优先使用后端数据：当用户切换设备或刷新页面时，从后端获取最新进度
       - 本地数据兜底：当网络不可用时，继续使用localStorage保存
       - 冲突检测：如果本地进度与后端进度不一致，提示用户选择使用哪个进度
       - 时间戳优先：如果无法确定优先级，以最后更新时间较新的进度为准
    """
    session = SessionLocal()
    try:
        # 查询用户是否已有活跃的测试会话
        existing_session = session.query(TestSession).filter(
            TestSession.user_id == user_id,
            TestSession.test_type == test_type,
            TestSession.session_status == 'active'
        ).first()
        
        if existing_session:
            # 检测冲突：比较当前设备与已有会话的设备标识
            if existing_session.device_info and device_info and existing_session.device_info != device_info:
                # 发生冲突，以最后更新时间为准
                if existing_session.updated_at > datetime.utcnow() - timedelta(minutes=5):
                    # 其他设备在5分钟内更新过，提示用户
                    return False, "检测到其他设备正在答题，已暂停当前设备的测试进度"
                else:
                    # 其他设备长时间未更新，使用当前设备的进度覆盖
                    existing_session.device_info = device_info
            
            # 更新现有会话
            existing_session.current_question = current_question
            existing_session.answers = answers
            existing_session.updated_at = datetime.utcnow()
            
            if device_info:
                existing_session.device_info = device_info
            
            session.commit()
            return True, "测试进度已更新"
        else:
            # 创建新的测试会话
            new_session = TestSession(
                user_id=user_id,
                test_type=test_type,
                current_question=current_question,
                total_questions=20,  # 默认值，实际应根据测试类型确定
                answers=answers,
                device_info=device_info,
                expires_at=datetime.utcnow() + timedelta(days=30)
            )
            
            session.add(new_session)
            session.commit()
            return True, "新的测试会话已创建"
    except Exception as e:
        session.rollback()
        return False, f"保存测试进度失败: {str(e)}"
    finally:
        session.close()

# 获取用户的活跃测试会话
def get_active_test_session(user_id, test_type):
    """
    获取用户的活跃测试会话
    
    参数:
        user_id: 用户ID
        test_type: 测试类型
    
    返回:
        TestSession对象或None
    """
    session = SessionLocal()
    try:
        active_session = session.query(TestSession).filter(
            TestSession.user_id == user_id,
            TestSession.test_type == test_type,
            TestSession.session_status == 'active',
            TestSession.expires_at > datetime.utcnow()
        ).first()
        return active_session
    finally:
        session.close()

# 结束测试会话
def complete_test_session(user_id, test_type):
    """
    结束测试会话，标记为已完成
    
    参数:
        user_id: 用户ID
        test_type: 测试类型
    """
    session = SessionLocal()
    try:
        session.query(TestSession).filter(
            TestSession.user_id == user_id,
            TestSession.test_type == test_type,
            TestSession.session_status == 'active'
        ).update({
            TestSession.session_status: 'completed',
            TestSession.updated_at: datetime.utcnow()
        })
        session.commit()
    finally:
        session.close()
  ```

#### 2.2.3 测试历史与分析
- **测试历史列表**：
  - 按时间倒序展示所有测试记录
  - 显示测试类型、日期、得分和状态
  - 支持按测试类型筛选
- **测试详情**：
  - 完整的测试结果报告
  - 得分分布图表
  - 因子分析（仅SCL-90）
  - 前后测试对比
  - 导出PDF报告
- **个人分析报告**：
  - 心理健康趋势图
  - 与同年级平均水平对比
  - 机器学习预测结果
  - 个性化建议

### 2.3 预警管理系统

#### 2.3.1 预警规则设计
- **预警触发机制**：基于心理测试结果的阈值触发
- **预警级别**：
  - **关注**：需要关注，无需紧急处理
  - **预警**：需要进一步关注，可能存在心理健康问题
- **内置预警规则**：
  | rule_name | trigger_condition | alert_level | trigger_source |
  |----------|----------|----------|----------|
  | 抑郁预警 | SDS标准分≥63分 | 预警 | test |
  | 焦虑预警 | SAS标准分≥60分 | 预警 | test |
  | 轻度关注 | SDS标准分≥53分或SAS标准分≥50分 | 关注 | test |
  | **情感抑郁预警** | 连续7个自然日内，每日至少1条情感分析记录，且每条记录均满足negative_score>positive_score且置信度>0.6 | 预警 | emotion |
  | **情感焦虑预警** | 连续5个自然日内，每日至少1条情感分析记录，且每条记录均满足anxiety_score>2且置信度>0.5 | 关注 | emotion |
  | **情绪波动预警** | 自然周内（周一至周日），情感记录≥5条，且情感得分标准差>0.8 | 关注 | emotion |

- **触发条件细节说明**：
  1. **时间计算方式**：
     - 所有基于时间的预警规则均使用"自然日"计算，即从当日0点到23点59分59秒为一个自然日
     - "连续N天"指连续的N个自然日，每个自然日内至少有一条符合条件的记录
     - "单周内"指当前自然周（周一至周日）
  
  2. **情感得分定义**（与EmotionAnalyzer保持一致）：
     - `negative_score`：负面情感得分，基于情感词典计算，值越高表示负面情感越强烈
     - `positive_score`：正面情感得分，基于情感词典计算，值越高表示正面情感越强烈
     - `anxiety_score`：焦虑情感得分，基于焦虑词典计算，值越高表示焦虑情绪越强烈
     - `置信度`：计算方法为`(abs(positive_score) + abs(negative_score)) / 有效词数`，取值范围0-1
  
  3. **情感预警实现逻辑**：
     ```python
     def check_emotion_alerts(user_id):
         """
         检查用户的情感预警条件
         
         参数:
             user_id: 用户ID
         
         返回:
             list: 满足的预警规则列表
         """
         alerts = []
         session = SessionLocal()
         
         try:
             # 检查情感抑郁预警：连续7天negative_score>positive_score且置信度>0.6
             seven_days_ago = datetime.utcnow() - timedelta(days=7)
             
             # 获取最近7天的情感分析记录
             emotion_records = session.query(Conversation).filter(
                 Conversation.user_id == user_id,
                 Conversation.timestamp >= seven_days_ago
             ).order_by(Conversation.timestamp).all()
             
             if emotion_records:
                 # 按自然日分组
                 daily_records = {}
                 for record in emotion_records:
                     # 转换为自然日（UTC时间）
                     date_key = record.timestamp.date()
                     if date_key not in daily_records:
                         daily_records[date_key] = []
                     daily_records[date_key].append(record)
                 
                 # 检查连续7天
                 consecutive_days = 0
                 today = datetime.utcnow().date()
                 
                 for day_offset in range(7):
                     check_date = today - timedelta(days=day_offset)
                     if check_date in daily_records:
                         # 检查当天是否有符合条件的记录
                         day_has_valid_record = any(
                             eval(record.emotion).get('negative_score', 0) > eval(record.emotion).get('positive_score', 0) and \
                             eval(record.emotion).get('置信度得分', 0) > 0.6
                             for record in daily_records[check_date]
                         )
                         if day_has_valid_record:
                             consecutive_days += 1
                         else:
                             break
                     else:
                         break
                 
                 if consecutive_days >= 7:
                     alerts.append('情感抑郁预警')
             
             return alerts
         finally:
             session.close()
     ```

#### 2.3.2.1 多重预警叠加处理
- **预警合并逻辑**：
  - 同一学生同时触发多个预警时，以最高级别为准
  - 不同类型预警分别记录，避免重复处理
  - 预警信息中包含所有触发的规则类型和原因
- **预警展示规则**：
  - 管理员端：合并显示同一学生的所有预警，按级别排序
  - 学生端：仅显示最严重的预警，避免信息过载
  - 预警统计：按类型和级别分别统计，支持多维度分析

#### 2.3.2 预警生成流程
1. **触发时机**：心理测试完成后立即触发
2. **预警生成**：
   - 记录预警信息到数据库
   - 更新学生心理状态
   - 生成预警统计数据
3. **预警通知**：
   - 系统内消息通知管理员
   - 管理员登录系统后可查看预警列表
4. **预警处理**：
   - 管理员可查看预警详情
   - 管理员可更新预警状态（待处理/已处理）
   - 管理员可添加处理记录

#### 2.3.3 预警统计分析
- **预警概览**：按级别、时间统计预警数量
- **预警趋势**：使用Python matplotlib/seaborn绘制预警数量随时间的变化趋势
- **群体特征**：分析预警学生的群体特征（年级、专业、性别等）
- **管理员视角**：管理员可查看所有班级的预警情况

### 2.4 数据统计与分析（Python增强版）

#### 2.4.1 基础统计分析（Python实现）
- **按年级统计**：使用Python pandas进行数据分组和聚合，分析各年级心理健康状况分布、预警数量
- **按专业统计**：分析各专业心理健康状况分布、压力源差异
- **按性别统计**：分析男女生心理健康状况差异
- **班级统计**：管理员可查看所有班级的详细统计数据

#### 2.4.2 Python机器学习预测模型
- **功能描述**：基于学生历史心理测试数据，使用Python机器学习算法预测未来心理健康风险
- **技术方案**：
  - **算法选择**：
    - 当训练数据≥500条时：XGBoost/Random Forest（Python scikit-learn/XGBoost库实现）
    - 当训练数据<500条时：基于专家规则的评分系统（避免小样本过拟合）
  - **特征工程**：使用Python pandas进行特征选择和数据预处理
  - **模型训练**：
    - 训练集：80%历史数据
    - 测试集：20%历史数据
    - 评估指标：准确率、召回率、F1-score
  - **输入特征**：历史测试得分、人口统计学信息（年级、专业、性别）
  - **输出**：未来3个月心理健康风险等级（低、中、高）
- **模型部署**：
  - 使用Python Flask RESTful API部署模型，支持实时预测
  - **模型版本管理**：
    - 实现模型版本号管理，记录模型训练时间、训练数据量、评估指标
    - 支持多版本模型共存，可随时切换默认模型
    - 实现模型自动更新机制，定期重新训练模型
  - **预测优化**：
    - 使用Redis缓存模型预测结果，提高响应速度
    - 实现模型预热机制，减少首次预测延迟
    - 支持批量预测，提高预测效率
- **可视化展示**：
  - 预测结果可视化（使用ECharts柱状图、雷达图）
  - 特征重要性分析（使用matplotlib绘制特征重要性图）
  - 预测准确率评估（使用混淆矩阵、ROC曲线）
  - 模型版本性能对比图

#### 2.4.3 交互式数据可视化（Python+ECharts）
- **核心功能**：
  - 支持多维度数据筛选和钻取
  - 实现自定义仪表盘功能
  - 添加动态筛选器，支持实时数据更新
  - 支持图表类型切换
- **特色可视化（Python实现）**：
  1. **班级心理状态热力图**
     - 按班级、学号维度展示学生心理状态分布
     - 使用Python生成热力图数据，ECharts可视化
     - 颜色编码：绿色（正常）→ 黄色（关注）→ 红色（预警）
     - 交互功能：鼠标悬停显示学生详情，点击查看测试历史
  
  2. **测试结果对比雷达图**
     - 学生个人测试结果与同专业均值的对比
     - 支持切换不同测试类型和对比群体
  
  3. **心理健康趋势预测图**
     - 展示学生未来心理健康风险预测结果
     - 结合历史测试数据，展示变化趋势
     - 使用Python机器学习模型生成预测数据
  
  4. **压力源关联网络图**
     - 展示压力源之间的关联关系
     - 使用Python networkx库生成关联网络数据
     - 使用networkx的spring_layout进行网络布局
     - ECharts可视化展示
     - 节点大小表示压力源出现频率
     - 边粗细表示关联强度

#### 2.4.4 Python异常测试行为检测
- **功能描述**：使用Python统计分析和聚类算法，分析学生测试行为，检测异常答题模式
- **检测指标**：
  - 答题时间异常（过快/过慢）
  - 答案一致性异常
  - 答题模式异常
- **实现方式**：
  - 使用Python pandas进行数据预处理
  - 使用Python scikit-learn DBSCAN算法进行异常检测
  - 设置异常阈值，自动标记异常测试
- **可视化展示**：使用Python matplotlib绘制异常检测结果

#### 2.4.5 Python个性化分析报告生成
- **功能描述**：使用Python自动生成学生个人心理健康分析报告
- **技术方案**：
  - 使用Python Jinja2模板引擎生成HTML报告
  - 使用Python matplotlib/seaborn生成报告中的图表
  - 支持导出为PDF格式（使用Python weasyprint库，HTML转PDF）
- **报告内容**：
  - 测试结果汇总
  - 心理健康趋势分析
  - Python机器学习预测结果
  - 个性化建议
  - 可视化图表
- **生成方式**：测试完成后自动生成，支持手动触发生成
- **权限控制**：学生可查看自己的报告，管理员可查看所有报告

#### 2.4.6 数据导出功能
- 支持使用Python pandas导出统计数据为Excel/PDF格式
- 支持导出ECharts图表为图片格式
- 支持自定义导出内容和范围

### 2.5 智能聊天机器人与情感分析模块

#### 2.5.1 智能心理健康助手
- **功能概述**：基于规则匹配的智能聊天机器人，为学生提供24/7心理健康支持
- **技术实现**：
  - **后端架构**：Python Flask RESTful API + SQLite知识库
  - **前端交互**：Bootstrap聊天界面 + WebSocket实时通信
  - **对话引擎**：基于规则的模式匹配 + 预设回复模板
- **核心对话功能**：
  - **心理健康问答**：
    - 焦虑相关问题的解答（如考试焦虑、社交焦虑等）
    - 抑郁情绪的识别和应对建议
    - 压力管理技巧和放松方法指导
    - 人际关系问题的建议
  - **测试流程指导**：
    - 心理测试前的准备工作指导
    - 测试过程中的问题解答
    - 测试结果的解释和后续建议
  - **资源推荐**：
    - 校园心理咨询联系方式
    - 推荐的心理学书籍和文章
    - 在线心理健康资源链接
  - **紧急情况处理**：
    - 危机情况的识别和引导
    - 紧急联系方式的快速提供
    - 专业求助渠道的推荐
- **对话管理**：
  - **对话历史记录**：保存学生与机器人的对话记录，便于管理员查看分析
  - **对话质量评估**：学生对机器人回答的满意度评价
  - **学习机制**：基于学生反馈不断优化回答质量
- **技术实现细节**：
  ```python
  # 聊天机器人核心类
  class MentalHealthBot:
      def __init__(self):
          self.knowledge_base = self.load_knowledge_base()
          self.conversation_context = {}
      
      def load_emotion_dict(self, file_path):
          """
          加载情感词典
          :param file_path: 词典文件路径
          :return: 情感词集合
          """
          with open(file_path, 'r', encoding='utf-8') as f:
              return set(f.read().splitlines())  # 假设每行一个词
      
      def load_stopwords(self, file_path):
          """
          加载停用词词典
          :param file_path: 停用词文件路径
          :return: 停用词集合
          """
          with open(file_path, 'r', encoding='utf-8') as f:
              return set(f.read().splitlines())  # 假设每行一个停用词
      
      def process_message(self, user_id, message):
          # 情感分析
          emotion = self.analyze_emotion(message)
          
          # 意图识别
          intent = self.recognize_intent(message)
          
          # 生成回复
          response = self.generate_response(intent, emotion, user_id)
          
          # 保存对话记录
          self.save_conversation(user_id, message, response, emotion)
          
          # 检查情感预警（智能机器人与预警系统联动）
          alerts = check_emotion_alerts(user_id)
          if alerts:
              # 根据预警类型生成不同的引导性回复
              for alert in alerts:
                  if alert == '情感抑郁预警':
                      response += "\n\n💡 我注意到你最近情绪可能比较低落，建议你可以尝试一些放松的活动，或者考虑使用系统的心理评估功能。"
                  elif alert == '情感焦虑预警':
                      response += "\n\n💡 我注意到你最近可能有一些焦虑情绪，建议你可以尝试深呼吸练习，或者使用系统的冥想引导功能。"
              
              # 记录预警触发事件
              log_system_operation(
                  operator_id=user_id,
                  operator_type='system',
                  operation_name='emotion_alert_triggered',
                  operation_details=json.dumps({"user_id": user_id, "alerts": alerts}),
                  execution_result='success'
              )
          
          return response
      
      def analyze_emotion(self, text):
          # 情感分析实现（支持否定词处理）
          positive_words = ['开心', '快乐', '满足', '愉快', '好', '幸福', '舒适', '安心']
          negative_words = ['焦虑', '抑郁', '压力', '烦恼', '难过', '痛苦', '担忧', '烦躁']
          negation_words = ['不', '没', '非', '无', '未', '勿', '不是', '没有']
          
          # 初始化情感得分和否定词标志
          score = 0
          negation_flag = False
          
          # 将文本分割为单词（简单分词，基于空格和常见标点）
          import re
          words = re.findall(r'[\w\u4e00-\u9fa5]+', text)
          
          for word in words:
              if word in negation_words:
                  negation_flag = not negation_flag
              elif word in positive_words:
                  score += -1 if negation_flag else 1
                  negation_flag = False  # 情感词后重置否定标志
              elif word in negative_words:
                  score += 1 if negation_flag else -1
                  negation_flag = False  # 情感词后重置否定标志
              # 非否定词也重置否定标志
              elif word not in negation_words:
                  negation_flag = False
          
          if score > 0:
              return 'positive'
          elif score < 0:
              return 'negative'
          else:
              return 'neutral'
      
      def save_conversation(self, user_id, user_message, bot_response, emotion):
          """
          保存对话记录到数据库
          
          参数:
              user_id: 用户ID
              user_message: 用户消息内容
              bot_response: 机器人回复内容
              emotion: 情感分析结果
          """
          from datetime import datetime
          from sqlalchemy import create_engine
          from sqlalchemy.orm import sessionmaker
          from sqlalchemy.ext.declarative import declarative_base
          from sqlalchemy import Column, Integer, String, Text, DateTime
          
          # SQLAlchemy配置（实际项目中应导入已配置的SessionLocal）
          Base = declarative_base()
          engine = create_engine('sqlite:///mental_health.db')
          SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
          
          # 定义Conversation模型
          class Conversation(Base):
              __tablename__ = 'conversation'
              
              id = Column(Integer, primary_key=True, autoincrement=True)
              user_id = Column(Integer, nullable=False)
              user_message = Column(Text, nullable=False)
              bot_response = Column(Text, nullable=False)
              emotion = Column(String(50), nullable=False)
              timestamp = Column(DateTime, default=datetime.utcnow)
          
          # 创建表
          Base.metadata.create_all(bind=engine)
          
          # 创建会话并保存记录
          db = SessionLocal()
          try:
              conversation = Conversation(
                  user_id=user_id,
                  user_message=user_message,
                  bot_response=bot_response,
                  emotion=emotion
              )
              db.add(conversation)
              db.commit()
              db.refresh(conversation)
          finally:
              db.close()
  ```

#### 2.5.2 情感分析引擎
- **功能概述**：分析学生输入文本的情感倾向，为预警系统提供辅助数据
- **技术实现**：
  - **算法选择**：基于情感词典的轻量级分析方法
  - **数据源**：学生聊天记录、测试反馈、情绪日记等文本内容
  - **分析维度**：正面/负面/中性情绪，焦虑/抑郁/压力等具体情感识别
- **情感分析功能**：
  - **实时情感识别**：对聊天消息进行实时情感分析
  - **情感趋势分析**：分析学生长期情感变化趋势
  - **群体情感统计**：统计不同群体（年级、专业）的整体情感状态
  - **情感预警**：当检测到持续的负面情感时触发预警
- **技术实现细节**：
  ```python
  import jieba
  from collections import Counter
  
  class EmotionAnalyzer:
      def __init__(self):
          # 加载中文情感词典
          self.positive_dict = self.load_emotion_dict('positive_words.txt')
          self.negative_dict = self.load_emotion_dict('negative_words.txt')
          self.anxiety_words = self.load_emotion_dict('anxiety_words.txt')
          self.depression_words = self.load_emotion_dict('depression_words.txt')
          # 加载停用词
          self.stopwords = self.load_stopwords('stopwords.txt')
      
      def analyze_text_emotion(self, text):
          # 输入验证
          if not text or len(text.strip()) == 0:
              return {
                  'primary_emotion': 'neutral',
                  'specific_emotions': {},
                  '置信度': 0.0
              }
          
          # 中文分词
          words = jieba.lcut(text)
          
          # 过滤停用词
          words = [word for word in words if word not in self.stopwords]
          
          # 防止空列表导致的除零错误
          if len(words) == 0:
              return {
                  'primary_emotion': 'neutral',
                  'specific_emotions': {},
                  '置信度': 0.0
              }
          
          # 处理否定词（如"不"、"没"等）
          negation_words = ['不', '没', '非', '无', '未', '勿']
          negation_flag = False
          processed_words = []
          
          for word in words:
              if word in negation_words:
                  negation_flag = not negation_flag
              else:
                  processed_words.append((word, negation_flag))
                  if word not in negation_words:
                      negation_flag = False
          
          # 计算情感得分，考虑否定词
          positive_score = 0
          negative_score = 0
          anxiety_score = 0
          depression_score = 0
          
          for word, has_negation in processed_words:
              if word in self.positive_dict:
                  positive_score += -1 if has_negation else 1
              if word in self.negative_dict:
                  negative_score += -1 if has_negation else 1
              if word in self.anxiety_words:
                  anxiety_score += -1 if has_negation else 1
              if word in self.depression_words:
                  depression_score += -1 if has_negation else 1
          
          # 情感分类
          if positive_score > negative_score:
              primary_emotion = 'positive'
          elif negative_score > positive_score:
              primary_emotion = 'negative'
          else:
              primary_emotion = 'neutral'
          
          # 具体情感识别
          specific_emotions = {}
          if anxiety_score > 0:
              specific_emotions['anxiety'] = anxiety_score
          if depression_score > 0:
              specific_emotions['depression'] = depression_score
          
          # 计算置信度，避免除零错误
          total_score = abs(positive_score) + abs(negative_score)
          置信度 = total_score / len(processed_words) if len(processed_words) > 0 else 0
          置信度 = min(置信度, 1.0)  # 限制置信度在0-1之间
          
          return {
              'primary_emotion': primary_emotion,
              'specific_emotions': specific_emotions,
              '置信度得分': 置信度,
              'positive_score': positive_score,
              'negative_score': negative_score,
              'word_count': len(processed_words)
          }
  ```python
- **数据模型设计**：
  ```sql
  -- 对话记录表
  CREATE TABLE conversation_log (
      id INTEGER PRIMARY KEY,
      user_id INTEGER NOT NULL,
      message TEXT NOT NULL,
      bot_response TEXT NOT NULL,
      emotion_score JSON, -- 存储结构化的情感得分数据
      emotion_type VARCHAR(20),
      satisfaction_score INTEGER,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES user(id)
  );
  
  -- 情感分析记录表
  CREATE TABLE emotion_analysis (
      id INTEGER PRIMARY KEY,
      user_id INTEGER NOT NULL,
      text_source VARCHAR(50), -- 'chat', 'diary', 'feedback'
      text_content TEXT NOT NULL,
      primary_emotion VARCHAR(20),
      emotion_details TEXT,
      analysis_date DATE,
      FOREIGN KEY (user_id) REFERENCES user(id)
  );
  ```

### 2.6 管理员功能模块（增强版）

#### 2.6.1 班级管理
- **班级学生管理**：
  - 查看所有班级学生列表和基本信息（姓名、学号、性别、年级、专业等）
  - 支持按班级、性别、状态筛选学生
  - 支持导出班级学生名单
- **班级心理状态概览**：
  - 查看班级学生整体心理状态分布（正常、关注、预警）
  - 查看班级平均测试得分和各因子得分分布
  - 班级心理状态热力图，直观展示每个学生的心理状态
  - **新增**：班级情感分析统计图表，展示学生群体情感变化
- **学生详情查看**：
  - 查看单个学生的完整心理测试历史记录
  - 查看学生预警信息和处理记录
  - 查看学生个性化分析报告和预测结果
  - 支持学生心理状态变化趋势图
  - **新增**：学生情感分析报告，包含聊天记录情感分析结果（仅展示脱敏后的内容，隐藏真实姓名）

#### 2.6.2 测试管理
- **测试任务发布**：
  - 选择测试类型（SCL-90/SAS/SDS）
  - 设置测试开始时间和截止时间
  - 选择发布范围（单个班级或多个班级）
  - 自定义测试说明和注意事项
  - 支持紧急测试发布
- **测试参与统计**：
  - 实时统计班级学生的测试参与情况（已完成、未完成、未开始）
  - 按完成时间排序展示学生测试进度
  - 支持提醒未完成测试的学生
- **测试结果分析**：
  - 班级测试结果汇总分析
  - 按因子或维度分析班级学生测试结果分布
  - 与同年级或全校平均水平对比分析
  - 生成班级测试结果雷达图和柱状图
  - **新增**：测试情感关联分析，分析测试得分与学生情感状态的相关性

#### 2.6.3 预警管理
- **预警列表**：
  - 按时间倒序展示所有学生的预警信息
  - 显示预警学生姓名、学号、预警类型、预警级别、预警时间
  - 支持按预警级别、预警类型、班级筛选
  - **新增**：情感预警标签，显示由情感分析触发的预警
- **预警处理**：
  - 查看预警详情和学生完整测试结果
  - 更新预警状态（待处理、已处理、已跟进）
  - 添加处理记录和备注
  - 支持批量处理预警
  - **新增**：查看预警学生的情感分析报告和聊天记录
- **预警统计**：
  - 统计预警数量和趋势
  - 分析预警学生的群体特征（性别、专业、年级等）
  - 生成预警统计报表
  - **新增**：情感预警触发统计，分析情感分析对预警系统的贡献度

#### 2.6.4 报告生成
- **班级心理报告**：
  - 自动生成班级整体心理健康分析报告
  - 包含班级心理状态分布、预警情况、测试结果分析等
  - 支持自定义报告周期（周/月/学期）
  - 生成可视化图表（热力图、柱状图、雷达图等）
- **学生个体报告**：
  - 查看或重新生成单个学生的心理分析报告
  - 包含测试结果、趋势分析、机器学习预测结果等
  - 支持添加管理员评语
- **报告导出**：
  - 支持导出班级报告为PDF/Excel格式
  - 支持导出多个学生报告为批量PDF
  - 支持自定义报告模板

#### 2.6.5 数据可视化
- **班级心理状态热力图**：按学生和时间维度展示心理状态变化
- **测试结果对比雷达图**：对比班级与同年级平均水平
- **心理健康趋势图**：展示班级整体心理健康变化趋势
- **预警分布饼图**：展示不同预警级别的分布情况
- **压力源关联网络图**：分析班级学生压力源的关联关系

### 2.7 学生端智能助手界面

#### 2.7.1 聊天机器人界面
- **聊天界面设计**：
  - **布局**：左侧聊天历史记录，右侧实时对话窗口
  - **消息类型**：用户消息、机器人回复、系统提示
  - **输入方式**：文本输入框，支持表情符号，字数限制500字
  - **响应反馈**：机器人回答后的满意度评价（👍/👎）
- **功能特性**：
  - **实时通信**：使用WebSocket实现实时消息传输
  - **消息历史**：保存最近30天的对话记录
  - **快速回复**：预设常用问题的快速回复按钮
  - **智能建议**：基于当前对话上下文提供相关问题建议
- **界面实现细节**：
  ```html
  <!-- 聊天界面结构 -->
  <div class="chat-container">
      <div class="chat-header">
          <h5>心理健康助手</h5>
          <span class="online-status">在线</span>
      </div>
      <div class="chat-messages" id="messageContainer">
          <!-- 消息显示区域 -->
      </div>
      <div class="chat-quick-replies">
          <button class="quick-reply-btn" data-message="我感到焦虑">我感到焦虑</button>
          <button class="quick-reply-btn" data-message="如何缓解压力">如何缓解压力</button>
          <button class="quick-reply-btn" data-message="想做心理测试">想做心理测试</button>
      </div>
      <div class="chat-input-area">
          <textarea id="messageInput" placeholder="输入你的问题..."></textarea>
          <button id="sendMessage">发送</button>
      </div>
  </div>
  ```
- **JavaScript实现**：
  ```javascript
  class ChatInterface {
      constructor() {
          this.ws = null;
          this.setupWebSocket();
          this.bindEvents();
      }
      
      setupWebSocket() {
          this.socket = io({
              reconnection: true,        // 启用重连
              reconnectionAttempts: 5,   // 最大重连尝试次数
              reconnectionDelay: 1000,   // 重连延迟（毫秒）
              reconnectionDelayMax: 5000, // 最大重连延迟（毫秒）
              timeout: 20000             // 连接超时时间（毫秒）
          });
          
          // 离线消息队列
          this.offlineMessages = [];
          // 用户在线状态
          this.isOnline = false;
          
          // 连接成功事件
          this.socket.on('connect', () => {
              console.log('WebSocket连接成功');
              this.isOnline = true;
              this.updateOnlineStatus('online');
              
              // 重连成功后，发送离线期间的消息
              if (this.offlineMessages.length > 0) {
                  this.offlineMessages.forEach(message => {
                      this.socket.emit('send_message', message);
                  });
                  this.offlineMessages = [];
              }
              
              // 重连成功后，请求未收到的消息
              this.socket.emit('get_missed_messages', {
                  user_id: currentUserId,
                  last_message_time: this.getLastMessageTime()
              });
          });
          
          // 连接断开事件
          this.socket.on('disconnect', () => {
              console.log('WebSocket连接断开');
              this.isOnline = false;
              this.updateOnlineStatus('offline');
          });
          
          // 重连尝试事件
          this.socket.on('reconnect_attempt', (attemptNumber) => {
              console.log(`正在尝试重连... (${attemptNumber}/5)`);
              this.updateOnlineStatus('reconnecting');
          });
          
          // 重连失败事件
          this.socket.on('reconnect_failed', () => {
              console.log('重连失败，请检查网络连接');
              this.updateOnlineStatus('offline');
          });
          
          // 接收消息事件
          this.socket.on('receive_message', (data) => {
              this.displayMessage(data.bot_response, 'bot');
              this.saveLastMessageTime(data.timestamp);
          });
          
          // 接收离线消息事件
          this.socket.on('missed_messages', (data) => {
              if (data.messages && data.messages.length > 0) {
                  data.messages.forEach(message => {
                      this.displayMessage(message.bot_response, 'bot');
                  });
                  this.saveLastMessageTime(data.messages[data.messages.length - 1].timestamp);
              }
          });
      }
      
      updateOnlineStatus(status) {
          const statusElement = document.querySelector('.online-status');
          if (statusElement) {
              statusElement.className = `online-status ${status}`;
              switch(status) {
                  case 'online':
                      statusElement.textContent = '在线';
                      break;
                  case 'offline':
                      statusElement.textContent = '离线';
                      break;
                  case 'reconnecting':
                      statusElement.textContent = '重连中...';
                      break;
              }
          }
      }
      
      getLastMessageTime() {
          // 从localStorage获取最后一条消息的时间
          return localStorage.getItem(`last_message_time_${currentUserId}`) || '1970-01-01T00:00:00';
      }
      
      saveLastMessageTime(timestamp) {
          // 将最后一条消息的时间保存到localStorage
          localStorage.setItem(`last_message_time_${currentUserId}`, timestamp);
      }
      
      sendMessage(message) {
          if (message.trim() === '') return;
          
          // 显示用户消息
          this.displayMessage(message, 'user');
          
          const messageData = {
              user_id: currentUserId,
              message: message
          };
          
          // 根据在线状态决定发送方式
          if (this.isOnline) {
              // 在线时直接发送
              this.socket.emit('send_message', messageData);
          } else {
              // 离线时将消息加入队列
              this.offlineMessages.push(messageData);
              // 显示离线提示
              this.displayMessage('当前处于离线状态，消息将在重新连接后发送', 'system');
          }
      }
      
      displayMessage(message, sender) {
          const container = document.getElementById('messageContainer');
          const messageDiv = document.createElement('div');
          messageDiv.className = `message ${sender}`;
          messageDiv.innerHTML = `
              <div class="message-content">${message}</div>
              <div class="message-time">${new Date().toLocaleTimeString()}</div>
          `;
          container.appendChild(messageDiv);
          container.scrollTop = container.scrollHeight;
      }
  }
  ```

#### 2.7.2 情感分析结果展示
- **个人情感报告页面**：
  - **情感趋势图**：展示最近7天/30天的情感变化趋势
  - **情感分布图**：饼图显示不同情感类型的占比
  - **关键词云**：展示学生常用的情感相关词汇
  - **情感建议**：基于情感分析结果提供个性化建议
- **数据可视化实现**：
  ```javascript
  // 情感趋势图表
  function renderEmotionTrend(data) {
      const chart = echarts.init(document.getElementById('emotionTrend'));
      const option = {
          title: { text: '情感变化趋势' },
          tooltip: { trigger: 'axis' },
          legend: { data: ['正面情绪', '负面情绪', '中性情绪'] },
          xAxis: { 
              type: 'category',
              data: data.dates 
          },
          yAxis: { type: 'value' },
          series: [
              {
                  name: '正面情绪',
                  type: 'line',
                  data: data.positive,
                  itemStyle: { color: '#5cb85c' }
              },
              {
                  name: '负面情绪',
                  type: 'line', 
                  data: data.negative,
                  itemStyle: { color: '#d9534f' }
              },
              {
                  name: '中性情绪',
                  type: 'line',
                  data: data.neutral,
                  itemStyle: { color: '#f0ad4e' }
              }
          ]
      };
      chart.setOption(option);
  }
  ```

### 2.8 系统管理

#### 2.8.1 用户管理
- 用户列表展示和搜索
- 用户创建、编辑、删除
- 角色权限管理

#### 2.8.2 数据管理
- 心理测试数据管理
- 预警数据管理
- **新增**：聊天记录管理和情感分析数据管理
- **新增**：对话质量统计分析
- 系统日志查看

#### 2.8.3 系统配置
- 基础配置（系统名称、Logo等）
- 测试类型和题目配置
- 预警规则配置
- **新增**：聊天机器人知识库管理
- **新增**：情感分析阈值配置
- **新增**：对话模板管理

#### 2.8.4 AI模块管理
- **聊天机器人配置**：
  - 知识库内容编辑和更新
  - 对话流程配置
  - 回复模板管理
- **情感分析配置**：
  - 情感词典管理
  - 分析参数调整
  - 预警阈值设置
- **AI性能监控**：
  - 对话成功率统计
  - 用户满意度监控
  - 情感分析准确率评估

### 2.9 实时更新功能

#### 2.9.1 功能概述
- **实时数据获取**：使用WebSocket + 轮询混合机制实现实时数据更新
- **实时更新内容**：测试结果、预警信息、班级心理状态、聊天消息、情感分析结果
- **更新频率**：
  - WebSocket：聊天消息、情感分析（实时）
  - 轮询：测试结果、预警信息（每5秒）
  - 轮询：统计数据、图表数据（每30秒）

#### 2.9.2 技术实现（Python + WebSocket + 轮询）
- **WebSocket实时通信**：
  - **服务器端**：Flask-SocketIO实现WebSocket服务
  - **客户端**：JavaScript Socket.IO客户端API
  - **应用场景**：聊天机器人、实时情感分析、紧急预警推送
- **轮询机制**：
  - **服务器端配置**：
    - 提供RESTful API接口，返回最新数据
    - 实现数据缓存机制，减少数据库查询频率
    - 设计高效的数据查询策略
  - **客户端配置**：
    - 实现定时请求机制
    - 处理服务器返回的最新数据
    - 更新页面内容和图表
- **WebSocket实现示例**：
  ```python
  # Flask-SocketIO实现
  from flask_socketio import SocketIO, emit, join_room, leave_room
  from datetime import datetime
  import sqlite3
  
  socketio = SocketIO(app, cors_allowed_origins="*")
  
  # 创建离线消息表
  def create_offline_messages_table():
      conn = sqlite3.connect('mental_health.db')
      cursor = conn.cursor()
      cursor.execute('''
          CREATE TABLE IF NOT EXISTS offline_messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id TEXT NOT NULL,
              message_type TEXT NOT NULL,
              content TEXT NOT NULL,
              timestamp DATETIME NOT NULL,
              delivered INTEGER DEFAULT 0
          )
      ''')
      conn.commit()
      conn.close()
  
  # 保存离线消息
  def save_offline_message(user_id, message_type, content):
      conn = sqlite3.connect('mental_health.db')
          cursor = conn.cursor()
          cursor.execute('''
              INSERT INTO offline_messages (user_id, message_type, content, timestamp)
              VALUES (?, ?, ?, ?)
          ''', (user_id, message_type, content, datetime.utcnow()))
          conn.commit()
          conn.close()
  
  # 获取离线消息
  def get_offline_messages(user_id, last_message_time):
      conn = sqlite3.connect('mental_health.db')
      cursor = conn.cursor()
      cursor.execute('''
          SELECT message_type, content, timestamp
          FROM offline_messages
          WHERE user_id = ? AND timestamp > ?
          ORDER BY timestamp ASC
      ''', (user_id, last_message_time))
      messages = cursor.fetchall()
      conn.close()
      return messages
  
  # 标记消息为已送达
  def mark_messages_as_delivered(user_id, last_message_time):
      conn = sqlite3.connect('mental_health.db')
      cursor = conn.cursor()
      cursor.execute('''
          UPDATE offline_messages
          SET delivered = 1
          WHERE user_id = ? AND timestamp <= ?
      ''', (user_id, last_message_time))
      conn.commit()
      conn.close()
  
  # 初始化离线消息表
  create_offline_messages_table()
  
  @socketio.on('join_chat')
  def on_join_chat(data):
      user_id = data['user_id']
      join_room(f'user_{user_id}')
      emit('status', {'msg': '已加入聊天室'})
  
  @socketio.on('send_message')
  def on_send_message(data):
      user_id = data['user_id']
      message = data['message']
      
      # 处理消息并生成回复
      bot_response = chat_bot.process_message(user_id, message)
      
      # 检查用户是否在线
      clients = socketio.server.manager.rooms.get(f'/user_{user_id}', {})
      is_online = len(clients) > 0
      
      # 使用UTC时间存储
      timestamp = datetime.utcnow().isoformat()
      response_data = {
          'bot_response': bot_response,
          'timestamp': timestamp
      }
      
      if is_online:
          # 用户在线，实时推送回复
          emit('receive_message', response_data, room=f'user_{user_id}')
      else:
          # 用户离线，保存为离线消息
          import json
          save_offline_message(user_id, 'receive_message', json.dumps(response_data))
  
  @socketio.on('emotion_analysis')
  def on_emotion_analysis(data):
      user_id = data['user_id']
      text = data['text']
      
      # 实时情感分析
      emotion_result = emotion_analyzer.analyze_text_emotion(text)
      
      # 检查用户是否在线
      clients = socketio.server.manager.rooms.get(f'/user_{user_id}', {})
      is_online = len(clients) > 0
      
      # 使用UTC时间存储
      timestamp = datetime.utcnow().isoformat()
      response_data = {
          'emotion': emotion_result,
          'timestamp': timestamp
      }
      
      if is_online:
          # 用户在线，实时推送分析结果
          emit('emotion_result', response_data, room=f'user_{user_id}')
      else:
          # 用户离线，保存为离线消息
          import json
          save_offline_message(user_id, 'emotion_result', json.dumps(response_data))
  
  @socketio.on('get_missed_messages')
  def on_get_missed_messages(data):
      user_id = data['user_id']
      last_message_time = data['last_message_time']
      
      # 获取离线消息
      offline_messages = get_offline_messages(user_id, last_message_time)
      
      # 转换为客户端需要的格式
      formatted_messages = []
      import json
      for message_type, content, timestamp in offline_messages:
          content_obj = json.loads(content)
          formatted_messages.append({
              'type': message_type,
              **content_obj
          })
      
      # 发送离线消息
      emit('missed_messages', {'messages': formatted_messages})
      
      # 标记消息为已送达
      if formatted_messages:
          latest_timestamp = formatted_messages[-1]['timestamp']
          mark_messages_as_delivered(user_id, latest_timestamp)
  ```

##### 2.9.3.1 聊天消息实时推送

##### 2.9.3.1 聊天消息实时推送
- **触发时机**：学生发送消息或机器人回复时
- **更新内容**：
  - 实时推送机器人回复消息
  - 显示消息发送状态和阅读状态
  - 更新对话满意度评价
- **实现方式**：
  - 使用Socket.IO建立持久连接
  - 消息通过Socket.IO实时推送
  - 客户端立即更新聊天界面

##### 2.9.3.2 情感分析实时反馈
- **触发时机**：学生发送聊天消息时
- **更新内容**：
  - 实时显示消息情感分析结果
  - 情感趋势图实时更新
  - 触发情感预警时立即通知
- **实现方式**：
  - Socket.IO推送情感分析结果
  - ECharts图表动态更新
  - 预警弹窗实时提醒

##### 2.9.3.3 测试结果实时更新
- **触发时机**：学生完成心理测试后
- **更新内容**：
  - 学生个人测试结果实时展示
  - 班级心理状态统计实时更新
  - 相关管理员收到测试完成通知
- **实现方式**：
  - 学生提交测试后，数据写入数据库
  - 管理员端页面定时请求最新测试结果
  - 页面自动更新测试完成通知

##### 2.9.3.4 预警信息实时更新
- **触发时机**：系统生成新的预警信息时
- **更新内容**：
  - 相关管理员收到预警通知
  - 预警管理页面更新预警列表
  - 班级心理状态热力图更新
- **实现方式**：
  - 系统检测到预警条件满足时，立即生成预警
  - 管理员端页面定时请求最新预警信息
  - 页面弹出预警提醒

##### 2.9.3.5 班级心理状态实时更新
- **触发时机**：班级内有学生完成测试或更新心理状态时
- **更新内容**：
  - 班级心理状态分布更新
  - 班级平均得分更新
  - 班级心理状态热力图刷新
- **实现方式**：
  - 前端定期请求班级心理状态统计数据
  - 使用ECharts的setOption方法更新图表
  - 实现平滑的数据过渡动画

##### 2.9.3.6 数据可视化图表实时刷新
- **触发时机**：相关数据发生变化时
- **更新内容**：
  - 班级心理状态热力图刷新
  - 测试结果对比雷达图更新
  - 心理健康趋势预测图更新
- **实现方式**：
  - 前端定期请求最新图表数据
  - 调用ECharts的setOption方法更新图表数据
  - 实现平滑的数据过渡动画

#### 2.9.4 轮询机制优化
- **数据缓存**：服务器端使用Redis缓存热点数据，减少数据库查询
- **增量更新**：只返回与上次请求相比变化的数据
- **动态更新频率**：根据数据重要性调整轮询间隔
- **请求合并**：将多个相关请求合并为一个，减少服务器负载

#### 2.9.5 轮询效果
- **响应时间**：关键数据更新响应时间 < 5秒
- **并发支持**：支持同时在线100+用户的轮询请求
- **资源占用**：服务器资源占用可控，适合毕设演示

#### 2.9.6 代码示例

**服务器端（Flask API）**：
```python
from flask import Flask, jsonify
import redis

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379, db=0)

# 获取最新预警信息
@app.route('/api/latest-alerts/<int:class_id>', methods=['GET'])
def get_latest_alerts(class_id):
    # 先从缓存获取，缓存不存在则从数据库查询
    cached_alerts = r.get(f'alerts:{class_id}')
    if cached_alerts:
        return jsonify(eval(cached_alerts))
    
    # 从数据库查询最新预警信息
    # 获取该班级的所有学生ID
    class_students = Student.query.filter_by(class_id=class_id).all()
    user_ids = [s.id for s in class_students]
    
    # 查询这些学生的预警信息
    alerts = Alert.query.filter(Alert.user_id.in_(user_ids)).order_by(Alert.created_at.desc()).limit(10).all()
    alerts_data = [alert.to_dict() for alert in alerts]
    
    # 缓存结果，设置过期时间5秒
    r.setex(f'alerts:{class_id}', 5, str(alerts_data))
    
    return jsonify(alerts_data)
```

**客户端（JavaScript）**：
```javascript
// 轮询获取最新预警信息
function pollLatestAlerts(classId) {
    fetch(`/api/latest-alerts/${classId}`)
        .then(response => response.json())
        .then(data => {
            // 更新预警列表
            updateAlertList(data);
            // 5秒后再次请求
            setTimeout(() => pollLatestAlerts(classId), 5000);
        })
        .catch(error => {
            console.error('轮询失败:', error);
            // 10秒后重试
            setTimeout(() => pollLatestAlerts(classId), 10000);
        });
}

// 页面加载时开始轮询
window.addEventListener('load', () => {
    const classId = document.getElementById('class-id').value;
    pollLatestAlerts(classId);
});
```



## 3. 非功能需求

### 3.1 性能需求（详细版）
- **页面加载性能**：
  - 首页加载时间 < 2秒
  - 测试页面加载时间 < 1.5秒
  - 数据可视化页面加载时间 < 3秒
  - 聊天页面加载时间 < 1秒
- **功能响应时间**：
  - 心理测试提交后结果生成 ≤ 3秒
  - 聊天机器人回复时间 < 2秒
  - 情感分析处理时间 < 1秒
  - 预警生成时间 < 1秒
- **数据处理性能**：
  - 简单数据库查询响应时间 < 500ms
  - 复杂统计分析查询响应时间 < 2秒
  - 机器学习预测响应时间 < 1.5秒
  - 数据导出处理时间 < 10秒
- **并发支持能力**：
  - 支持同时在线用户数 ≥ 100人
  - WebSocket连接数 ≥ 80个
  - 并发聊天消息处理 ≥ 50条/秒
  - 数据库连接池 ≥ 20个连接
- **系统资源要求**：
  - 内存占用 < 2GB（正常运行状态）
  - CPU使用率 < 70%（正常运行状态）
  - 磁盘空间需求 < 10GB（包含数据和日志）

### 3.2 安全性需求（简化版）
- **数据安全**：
  - 密码使用bcrypt加密存储
  - 敏感数据脱敏处理：
    - 姓名脱敏：只显示姓氏，名字用*替换（如：李**）
    - 学号脱敏：显示前4位和后2位，中间用*替换（如：2022****01）
    - 联系电话脱敏：显示前3位和后4位，中间用*替换（如：138****1234）
    - 不使用假名生成，保留真实数据的统计意义
  - 数据导出时加密：
    - 导出的Excel/PDF文件支持密码保护
    - 敏感数据在导出文件中同样进行脱敏处理
- **应用安全**：防止SQL注入、XSS攻击、CSRF攻击
- **认证与授权**：基于角色的访问控制，强密码策略

### 3.3 可用性需求（简化版）
- **系统可用性**：≥ 98%
- **界面兼容性**：支持主流浏览器（Chrome、Edge、Firefox）
- **操作容错**：表单数据自动保存，关键操作二次确认

### 3.4 可维护性需求
- **代码规范**：统一的代码风格，代码注释覆盖率 ≥ 30%
- **文档要求**：系统架构文档、数据库设计文档、API文档

## 4. 技术栈选择

| category | technology | version | purpose |
|------|------|------|------|
| **前端框架** | Bootstrap | 5.3 | 响应式UI框架 |
| | ECharts | 5.4 | 数据可视化图表库 |
| | Jinja2 | 3.1 | 模板引擎 |
| | 原生JavaScript | - | DOM操作和AJAX |
| **后端框架** | Flask | 2.3 | Web应用框架 |
| | SQLAlchemy | 2.0 | ORM框架 |
| | Flask-Login | 0.6 | 认证和授权 |
| | Flask-Bcrypt | 1.0 | 密码加密 |
| **数据库** | SQLite | 3.30+ | 开发环境/毕设演示 |
| | MySQL | 8.0 | 可选生产环境 |
| | Redis | 7.0+ | 缓存服务 |
| **AI模块** | scikit-learn | 1.3 | 机器学习算法 |
| | XGBoost | 2.0 | 梯度提升算法 |
| | pandas | 2.0 | 数据处理 |
| | numpy | 1.24 | 数值计算 |
| | networkx | 3.1 | 网络分析与布局 |
| | jieba | 0.42 | 中文分词工具 |
| | weasyprint | 62.1 | HTML转PDF工具 |
| **实时通信** | Flask-SocketIO | 5.3 | WebSocket实时通信 |
| **开发工具** | Git | 2.42 | 版本控制 |
| | PyCharm | 2023.2 | IDE |
| | Postman | 10.21 | API测试 |

## 5. 数据模型设计（简化版）

### 5.1 实体关系图（ER图）
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     User        │    │    Student      │    │     Class       │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ id (PK)        │◄──►│ id (PK)         │◄──►│ id (PK)         │
│ username        │    │ user_id (FK)    │    │ class_name      │
│ password_hash   │    │ student_id      │    │ grade          │
│ role           │    │ name            │    │ major          │
│ status         │    │ grade           │    │ student_count  │
│ created_at     │    │ major           │    │ created_at     │
└─────────────────┘    │ gender          │    └─────────────────┘
                       │ class_id (FK)   │
                       │ phone           │
                       │ created_at      │
                       └─────────────────┘
                                │
                                │
                       ┌─────────────────┐
                       │PsychologicalTest│
                       ├─────────────────┤
                       │ id (PK)         │◄──
                       │ user_id (FK)     │
                       │ test_type       │
                       │ test_date       │
                       │ status          │
                       │ raw_score      │
                       │ standard_score │
                       │ result         │
                       │ created_at     │
                       │ completed_at   │
                       └─────────────────┘
                                │
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   TestResult   │    │     Alert       │
                       ├─────────────────┤    ├─────────────────┤
                       │ id (PK)         │    │ id (PK)         │◄──
                       │ test_id (FK)    │    │ user_id (FK)     │
                       │ factor_name     │    │ alert_type      │
                       │ factor_score    │    │ alert_level    │
                       │ factor_level    │    │ trigger_source  │
                       │ item_answers    │    │ trigger_value  │
                       │ created_at      │    │ description    │
                       └─────────────────┘    │ status         │
                                              │ handler_id     │
                                              │ handle_note    │
                                              │ created_at     │
                                              │ processed_at   │
                                              └─────────────────┘
                                │
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │ConversationLog │    │EmotionAnalysis │
                       ├─────────────────┤    ├─────────────────┤
                       │ id (PK)         │    │ id (PK)         │◄──
                       │ user_id (FK)    │    │ user_id (FK)    │
                       │ session_id       │    │ text_source     │
                       │ message         │    │ text_content   │
                       │ bot_response    │    │ primary_emotion│
                       │ emotion_score   │    │ emotion_details│
                       │ satisfaction   │    │ 置信度         │
                       │ response_time   │    │ word_count     │
                       │ created_at      │    │ keywords       │
                       └─────────────────┘    │ analysis_date  │
                                              │ created_at     │
                                              └─────────────────┘
```

### 5.2 核心实体关系
- **用户（User）**：基础用户信息
- **学生（Student）**：学生详细信息（与User一对一）
- **心理测试（PsychologicalTest）**：测试记录（与Student一对多）
- **预警（Alert）**：预警记录（与Student一对多）
- **测试结果（TestResult）**：测试得分详情（与PsychologicalTest一对一）
- **班级（Class）**：班级信息（与Student一对多）
- **对话记录（ConversationLog）**：聊天记录（与Student一对多）
- **情感分析（EmotionAnalysis）**：情感分析结果（与Student一对多）

### 5.3 数据字典
#### 枚举类型定义
- **用户角色（user.role）**：student（学生）、admin（管理员）
- **用户状态（user.status）**：active（活跃）、inactive（禁用）
- **测试状态（psychological_test.status）**：in_progress（进行中）、completed（已完成）
- **预警级别（alert.alert_level）**：attention（关注）、alert（预警）
- **预警状态（alert.status）**：pending（待处理）、processed（已处理）、followed_up（已跟进）
- **情感类型（emotion_analysis.primary_emotion）**：positive（正面）、negative（负面）、neutral（中性）

### 5.3 主要表结构

#### 用户表（user）
| 字段名 | 数据类型 | 描述 |
|--------|----------|------|
| id | int | 主键 |
| username | varchar(50) | 用户名 |
| password_hash | varchar(255) | 密码哈希 |
| role | enum | 角色（student/admin） |
| status | enum | 状态（active/inactive） |
| created_at | datetime | 创建时间 |

#### 学生表（student）
| 字段名 | 数据类型 | 描述 |
|--------|----------|------|
| id | int | 主键 |
| user_id | int | 外键（关联user表） |
| student_id | varchar(20) | 学号 |
| name | varchar(50) | 姓名 |
| grade | varchar(10) | 年级 |
| major | varchar(50) | 专业 |
| class_id | int | 外键（关联class表） |
| gender | varchar(10) | 性别 |
| phone | varchar(20) | 联系电话（可选） |



#### 班级表（class）
| 字段名 | 数据类型 | 描述 |
|--------|----------|------|
| id | int | 主键 |
| class_name | varchar(20) | 班级名称 |
| grade | varchar(10) | 年级 |
| major | varchar(50) | 专业 |
| student_count | int | 学生人数 |
| created_at | datetime | 创建时间 |

#### 心理测试表（psychological_test）
| 字段名 | 数据类型 | 描述 |
|--------|----------|------|
| id | int | 主键 |
| user_id | int | 外键（关联user表） |
| test_type | enum | 测试类型（SCL-90/SAS/SDS） |
| test_date | datetime | 测试日期 |
| status | enum | 状态（completed/in_progress） |
| raw_score | float | 原始得分 |
| standard_score | float | 标准得分 |
| result | varchar(50) | 测试结果 |
| created_at | datetime | 创建时间 |
| completed_at | datetime | 完成时间 |

#### 测试结果表（test_result）
| 字段名 | 数据类型 | 描述 |
|--------|----------|------|
| id | int | 主键 |
| test_id | int | 外键（关联psychological_test表） |
| factor_name | varchar(50) | 因子名称（如：焦虑、抑郁等） |
| factor_score | float | 因子得分 |
| factor_level | varchar(20) | 因子等级（低/中/高） |
| item_answers | json | 具体题目答案（JSON格式） |
| created_at | datetime | 创建时间 |

#### 预警表（alert）
| 字段名 | 数据类型 | 描述 |
|--------|----------|------|
| id | int | 主键 |
| user_id | int | 外键（关联user表） |
| alert_type | varchar(50) | 预警类型（anxiety/depression/attention等） |
| alert_level | enum | 预警级别（attention/alert） |
| trigger_source | varchar(50) | 触发源（test/emotion） |
| trigger_value | float | 触发值 |
| description | text | 预警描述 |
| status | enum | 状态（pending/processed/followed_up） |
| handler_id | int | 处理人ID（外键关联user表） |
| handle_note | text | 处理备注 |
| created_at | datetime | 创建时间 |
| processed_at | datetime | 处理时间 |

#### 对话记录表（conversation_log）
| 字段名 | 数据类型 | 描述 |
|--------|----------|------|
| id | int | 主键 |
| user_id | int | 外键（关联user表） |
| session_id | varchar(100) | 会话ID |
| message | text | 用户消息内容 |
| bot_response | text | 机器人回复内容 |
| emotion_score | json | 情感得分（JSON格式） |
| emotion_type | varchar(20) | 情感类型（positive/negative/neutral） |
| satisfaction_score | tinyint | 满意度评分（1-5分） |
| response_time_ms | int | 响应时间（毫秒） |
| created_at | datetime | 创建时间 |

#### 情感分析表（emotion_analysis）
| 字段名 | 数据类型 | 描述 |
|--------|----------|------|
| id | int | 主键 |
| user_id | int | 外键（关联user表） |
| text_source | varchar(50) | 文本来源（chat/diary/feedback） |
| text_content | text | 原始文本内容 |
| primary_emotion | varchar(20) | 主要情感类型 |
| emotion_details | json | 详细情感分析结果 |
| confidence_score | float | 分析置信度 |
| word_count | int | 文本字数 |
| keywords | varchar(255) | 关键词提取 |
| analysis_date | date | 分析日期 |
| created_at | datetime | 创建时间 |

#### AI知识库表（ai_knowledge_base）
| 字段名 | 数据类型 | 描述 |
|--------|----------|------|
| id | int | 主键 |
| category | varchar(50) | 知识分类（anxiety/depression/stress等） |
| question_pattern | text | 问题匹配模式（正则表达式） |
| answer_template | text | 回答模板 |
| keywords | varchar(255) | 关键词 |
| priority | int | 优先级（数字越大优先级越高） |
| status | enum | 状态（active/inactive） |
| usage_count | int | 使用次数统计 |
| created_at | datetime | 创建时间 |
| updated_at | datetime | 更新时间 |

### 5.4 API接口设计

#### 5.4.1 认证相关接口
- **POST /api/auth/login**：用户登录
  ```json
  // 请求
  {
    "username": "student001",
    "password": "password123"
  }
  // 响应
  {
    "success": true,
    "user_id": 123,
    "username": "student001",
    "role": "student",
    "token": "jwt_token_here"
  }
  ```

- **POST /api/auth/logout**：用户注销
- **GET /api/auth/profile**：获取当前用户信息

#### 5.4.2 心理测试相关接口
- **GET /api/tests**：获取测试类型列表
- **POST /api/tests/start**：开始测试
- **POST /api/tests/submit**：提交测试答案
- **GET /api/tests/history**：获取测试历史
- **GET /api/tests/{id}/result**：获取测试结果

#### 5.4.3 聊天机器人接口
- **WebSocket /chat**：实时聊天通信
- **POST /api/chat/message**：发送消息（HTTP方式）
- **GET /api/chat/history**：获取聊天历史

#### 5.4.4 情感分析接口
- **POST /api/emotion/analyze**：文本情感分析
- **GET /api/emotion/trend**：获取情感趋势数据
- **GET /api/emotion/report**：获取情感分析报告

#### 5.4.5 管理员接口
- **GET /api/admin/users**：用户管理
- **GET /api/admin/alerts**：预警管理
- **GET /api/admin/statistics**：数据统计
- **GET /api/admin/knowledge**：AI知识库管理

## 6. 系统实施计划（毕设版）

### 6.1 开发阶段（详细版）

#### 第1-2周：需求分析与准备阶段
- **第1周**：
  - 需求细化分析，明确各角色功能边界
  - 学习核心技术栈（Flask、SQLAlchemy、Bootstrap）
  - 搭建开发环境，配置IDE、数据库
  - 编写系统架构设计文档
- **第2周**：
  - 数据库详细设计，绘制ER图
  - 设计系统API接口
  - 设计前端页面原型
  - 准备Kaggle数据集获取与处理方案
  - 编写数据库设计文档

#### 第3-4周：核心功能实现阶段
- **第3周**：
  - 实现用户认证系统（登录、注册、注销）
  - 实现RBAC权限控制基础框架
  - 实现用户管理功能（管理员）
  - 实现数据库表结构迁移
- **第4周**：
  - 实现心理测试基础功能（测试类型管理、题目管理）
  - 实现测试流程（测试开始、答题、提交、结果计算）
  - 实现测试进度保存机制
  - 实现学生端测试页面
  - **新增**：实现情感分析基础功能（文本预处理、情感词典加载）

#### 第5-6周：测试结果与预警管理阶段
- **第5周**：
  - 实现测试结果分析功能
  - 实现测试历史记录管理
  - 实现个人分析报告生成基础功能
  - 实现学生端测试历史与报告页面
- **第6周**：
  - 实现预警规则设计与配置
  - 实现预警生成与处理流程
  - 实现预警通知机制
  - 实现管理员端预警管理页面

#### 第7-8周：数据统计与机器学习阶段
- **第7周**：
  - 实现基础统计分析功能（按年级、专业、班级统计）
  - 实现Kaggle数据集爬取、清洗、集成功能
  - 实现数据可视化基础框架
  - 实现管理员端数据统计页面
- **第8周**：
  - 实现机器学习预测模型（XGBoost/Random Forest）
  - 实现模型训练、评估、部署
  - 实现模型API接口
  - 实现预测结果可视化

#### 第9-10周：AI模块与交互式可视化实现阶段
- **第9周**：
  - 实现智能聊天机器人功能（规则匹配、知识库管理）
  - 实现WebSocket实时通信功能
  - 实现学生端聊天界面
  - 实现AI知识库管理界面
  - **新增**：集成情感分析到聊天流程
- **第10周**：
  - 实现交互式数据可视化功能（ECharts）
  - 实现班级心理状态热力图
  - 实现测试结果对比雷达图
  - 实现心理健康趋势预测图
  - 实现压力源关联网络图
  - **新增**：实现情感分析结果可视化图表

#### 第11-12周：AI模块优化与系统测试阶段
- **第11周**：
  - 完善聊天机器人功能和对话质量
  - 优化情感分析算法准确性
  - 实现实时更新功能完善
  - 实现个性化分析报告自动生成
  - 实现报告导出功能（PDF/Excel）
  - 实现管理员端AI模块管理界面
- **第12周**：
  - 单元测试与集成测试
  - AI模块专项测试和优化
  - 性能测试与优化
  - 安全性测试与修复
  - 用户体验测试与优化
  - 完善系统文档（API文档、使用手册）
  - **新增**：编写AI模块技术文档和测试报告

#### 第13-14周：答辩准备与演示阶段
- **第13周**：
  - 完成毕设论文撰写
  - 准备答辩PPT
  - 系统部署与演示环境搭建
  - 进行系统演示预演
- **第14周**：
  - 答辩准备与练习
  - 最终系统优化
  - 提交毕设论文与系统
  - 进行毕设答辩

### 6.2 重点功能优先级
1. **核心功能**：心理测试、测试结果分析、数据管理
2. **AI特色功能**：智能聊天机器人、情感分析引擎、机器学习预测模型
3. **可视化功能**：交互式数据可视化、实时图表更新
4. **辅助功能**：预警管理、个性化报告生成
5. **增强功能**：实时更新、Kaggle数据集集成

### 6.3 技术实施要点
- **代码管理**：使用Git进行版本控制，定期提交代码
- **文档编写**：同步编写技术文档，保持文档与代码一致性
- **测试驱动**：采用单元测试+集成测试的测试策略
- **性能优化**：重点优化数据库查询和页面加载速度
- **安全性**：实现输入验证、防止SQL注入、XSS攻击等安全措施

### 6.4 风险控制
- **技术风险**：针对不熟悉的技术，提前学习和练习
- **时间风险**：预留缓冲时间，应对可能的延期
- **数据风险**：做好数据备份，防止数据丢失
- **质量风险**：严格按照需求和设计文档开发，避免功能偏差

## 7. 毕设特色亮点

1. **智能心理健康助手**：基于规则匹配的AI聊天机器人，为学生提供24/7心理健康支持，集成情感分析功能实现智能情感识别
2. **实时情感分析引擎**：使用中文分词和情感词典技术，实时分析学生输入文本的情感倾向，为预警系统提供辅助数据
3. **WebSocket实时通信**：实现聊天消息、情感分析结果、预警信息的实时推送，响应时间<1秒，支持100+并发用户
4. **多维度AI分析**：结合机器学习预测模型、情感分析引擎、智能问答系统，为学生提供全方位的心理健康分析
5. **交互式数据可视化**：实现多维度数据筛选、钻取和自定义仪表盘，支持班级心理状态热力图、测试结果对比雷达图、情感趋势图等多种可视化形式
6. **Kaggle数据集集成**：实现自动化的Kaggle数据集爬取、清洗、预处理和集成，丰富系统数据来源，提高分析模型的准确性
7. **机器学习预测**：使用XGBoost算法预测学生未来心理健康风险，结合情感分析数据动态调整预测结果
8. **个性化分析报告**：自动生成包含测试结果、情感分析、预测结果和个性化建议的综合分析报告，支持PDF/Excel导出
9. **针对2022级大学生的定制化测试**：添加压力源评估量表和疫情后心理健康恢复量表，重点关注当代大学生的心理健康需求
10. **现代化技术栈**：采用Flask-SocketIO、jieba、ECharts 5等现代技术，展示AI技术在心理健康领域的应用

## 8. 系统架构与部署

### 8.1 系统架构图
```
┌─────────────────────────────────────────────────────────────────────┐
│                        前端层 (Frontend)                      │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│ │  学生端    │  │  管理员端  │  │  聊天界面  │        │
│ ├─────────────┤  ├─────────────┤  ├─────────────┤        │
│ │ Bootstrap 5 │  │ Bootstrap 5 │  │ Bootstrap 5 │        │
│ │ ECharts 5  │  │ ECharts 5  │  │ Socket.IO   │        │
│ │ JavaScript  │  │ JavaScript  │  │ JavaScript  │        │
│ └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────────────┐
│                     应用层 (Application Layer)                │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │                Flask Web Framework                     │  │
│ ├─────────────────────────────────────────────────────────────┤  │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │  │
│ │ │ 认证模块   │ │ 权限控制   │ │ 业务逻辑   │ │  │
│ │ └─────────────┘ └─────────────┘ └─────────────┘ │  │
│ └─────────────────────────────────────────────────────────────┘  │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │               Socket.IO 实时通信                     │  │
│ └─────────────────────────────────────────────────────────────┘  │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │                   AI 模块                          │  │
│ ├─────────────────────────────────────────────────────────────┤  │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │  │
│ │ │聊天机器人   │ │ 情感分析   │ │机器学习模型 │ │  │
│ │ └─────────────┘ └─────────────┘ └─────────────┘ │  │
│ └─────────────────────────────────────────────────────────────┘  │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │              RESTful API 接口                     │  │
│ └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼ SQL/NoSQL
┌─────────────────────────────────────────────────────────────────────┐
│                     数据层 (Data Layer)                     │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│ │   SQLite   │  │    MySQL   │  │   Redis    │        │
│ │(开发环境)   │  │ (生产环境)   │  │   (缓存)   │        │
│ └─────────────┘  └─────────────┘  └─────────────┘        │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│ │ 用户数据    │  │ 测试数据    │  │ 知识库数据  │        │
│ │ 预警数据    │  │ 情感数据    │  │ 日志数据    │        │
│ └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 系统流程图

#### 8.2.1 心理测试流程图
```
学生登录系统
        │
        ▼
┌─────────────────┐
│  选择测试类型   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  测试说明展示   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   开始答题     │─────────┐
└─────────────────┘         │
        │                 │
        ▼                 │
┌─────────────────┐         │
│  逐题作答     │         │
│ (实时保存进度)  │         │
└─────────────────┘         │
        │                 │
        ▼                 │
┌─────────────────┐         │
│   提交测试     │         │
└─────────────────┘         │
        │                 │
        ▼                 │
┌─────────────────┐         │
│  生成测试结果   │         │
└─────────────────┘         │
        │                 │
        ▼                 │
┌─────────────────┐         │
│  生成预警信息   │         │
└─────────────────┘         │
        │                 │
        ▼                 │ 预警条件满足
┌─────────────────┐         │ ┌─────────────────┐
│  显示测试报告   │         │ │ 通知管理员     │
└─────────────────┘         │ └─────────────────┘
        │                 │
        ▼                 │
┌─────────────────┐         │
│  保存数据记录   │◄────────┘
└─────────────────┘
```

#### 8.2.2 AI聊天机器人流程图
```
学生发送消息
        │
        ▼
┌─────────────────┐
│  消息预处理   │ (去除特殊字符、长度检查)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  情感分析     │ (计算情感倾向)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  意图识别     │ (正则表达式匹配)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  查询知识库   │ (规则匹配)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  生成回复内容   │ (模板填充)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  保存对话记录   │ (情感分析+回复)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  实时推送给学生 │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  学生满意度评价 │
└─────────────────┘
```

#### 8.2.3 预警处理流程图
```
心理测试完成
        │
        ▼
┌─────────────────┐
│  计算测试得分   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  检查预警规则   │
└─────────────────┘
        │
        ▼ (规则匹配)
┌─────────────────┐
│  生成预警记录   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  更新学生状态   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  推送通知消息   │
└─────────────────┘
        │
        ▼
┌─────────────────┐     ┌─────────────────┐
│  管理员查看     │────►│  管理员处理     │
└─────────────────┘     └─────────────────┘
        │                       │
        ▼ (30天内未处理)         ▼ (处理完成)
┌─────────────────┐         ┌─────────────────┐
│  自动升级预警   │         │  更新预警状态   │
└─────────────────┘         └─────────────────┘
```

### 8.5 部署方案
#### 8.5.1 开发环境部署
- **本地开发**：
  - Python 3.9+ 虚拟环境
  - SQLite 数据库
  - Flask 开发服务器
  - 前端静态文件服务

#### 8.5.2 生产环境部署
- **推荐部署架构**：
  - **Web服务器**：Nginx（反向代理 + 静态文件服务）
  - **应用服务器**：Gunicorn（WSGI服务器）
  - **数据库**：MySQL 8.0
  - **缓存服务**：Redis 7.0
  - **容器化**：Docker（可选）

### 8.3 AI模块流程图

#### 8.3.1 智能聊天机器人模块流程图
```
┌─────────────────────────────────────────────────────────────────────┐
│                    智能聊天机器人架构                        │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│ │ 消息接收层  │───►│ 意图理解层  │───►│ 回复生成层  │  │
│ └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                   │                   │         │
│        ▼                   ▼                   ▼         │
│ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│ │Socket.IO通信│    │ 规则匹配引擎│    │ 模板引擎    │  │
│ │消息解析    │    │ 正则表达式   │    │动态填充    │  │
│ │身份验证    │    │ 关键词提取   │    │个性化处理   │  │
│ └─────────────┘    └─────────────┘    └─────────────┘  │
│                          │                   │         │
│                          ▼                   ▼         │
│                 ┌─────────────┐    ┌─────────────┐  │
│                 │ 知识库查询 │    │ 情感分析   │  │
│                 │ 优先级排序  │    │ 情绪识别   │  │
│                 │ 模板匹配   │    │ 信心度计算  │  │
│                 └─────────────┘    └─────────────┘  │
│                          │                   │         │
│                          └─────────┬─────────┘         │
│                                    ▼                   │
│                          ┌─────────────────────┐  │
│                          │  智能决策引擎     │  │
│                          ├─────────────────────┤  │
│                          │ 基于情感调整语气  │  │
│                          │ 根据历史个性化  │  │
│                          │ 紧急情况识别    │  │
│                          └─────────────────────┘  │
│                                    ▼                   │
│                          ┌─────────────────────┐  │
│                          │   回复输出生成    │  │
│                          ├─────────────────────┤  │
│                          │ 格式化回复内容    │  │
│                          │ 添加相关建议链接  │  │
│                          │ 生成学习数据    │  │
│                          └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 8.3.2 情感分析引擎流程图
```
┌─────────────────────────────────────────────────────────────────────┐
│                   情感分析引擎架构                            │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│ │ 文本预处理  │───►│ 特征提取层  │───►│ 情感计算层  │  │
│ └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                   │                   │         │
│        ▼                   ▼                   ▼         │
│ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│ │中文分词    │    │ 情感词典   │    │ 多维度评分  │  │
│ │jieba分词   │    │ 正面词汇库  │    │ 权重计算   │  │
│ │去停用词    │    │ 负面词汇库  │    │ 因子分解   │  │
│ │文本清洗    │    │ 焦虑词汇库  │    │ 上下文分析 │  │
│ └─────────────┘    │ 抑郁词汇库  │    └─────────────┘  │
│                   └─────────────┘           │         │
│                          │                   ▼         │
│                          ▼                   │         │
│                 ┌─────────────────────┐         │         │
│                 │   词典匹配引擎     │         │         │
│                 ├─────────────────────┤         │         │
│                 │ 精确字符串匹配    │         │         │
│                 │ 模糊匹配算法    │         │         │
│                 │ 同义词扩展      │         │         │
│                 │ 否定句处理      │         │         │
│                 └─────────────────────┘         │         │
│                          │                   │         │
│                          ▼                   ▼         │
│                 ┌─────────────┐    ┌─────────────┐  │
│                 │ 情感聚合   │    │ 置信度计算  │  │
│                 │ 加权平均   │    │ 一致性检验  │  │
│                 │ 极值处理   │    │ 阈值判断   │  │
│ │ 中性情绪平衡│    │ 模糊处理   │  │
│ └─────────────┘    └─────────────┘  │
│                          │                   │         │
│                          ▼                   ▼         │
│                 ┌─────────────────────┐         │         │
│                 │   结果输出生成     │         │         │
│                 ├─────────────────────┤         │         │
│                 │ 主导情感分类    │         │         │
│                 │ 具体情感得分    │         │         │
│                 │ 情感强度等级    │         │         │
│                 │ 关键词提取      │         │         │
│                 └─────────────────────┘         │         │
└─────────────────────────────────────────────────────────────┘
```

#### 8.3.3 机器学习预测模块流程图
```
┌─────────────────────────────────────────────────────────────────────┐
│                 机器学习心理健康预测系统                        │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│ │ 数据预处理  │───►│  模型训练   │───►│  预测服务   │  │
│ └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                   │                   │         │
│        ▼                   ▼                   ▼         │
│ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│ │ 特征工程   │    │ 算法选择   │    │ 实时预测   │  │
│ │标准化处理  │    │ XGBoost/RF │    │ 模型加载   │  │
│ │特征选择   │    │ 逻辑回归    │    │ API接口    │  │
│ │编码转换   │    │ 模型评估   │    │ 批量处理   │  │
│ │缺失值处理  │    │ 交叉验证   │    │ 缓存优化   │  │
│ └─────────────┘    └─────────────┘    └─────────────┘  │
│                          │                   │         │
│                          ▼                   ▼         │
│                 ┌─────────────┐    ┌─────────────┐  │
│                 │ 模型部署   │    │ 性能监控   │  │
│ │版本管理   │    │ 准确率跟踪 │  │
│ │A/B测试   │    │ 漂移检测   │  │
│ │自动重训   │    │ 反馈收集   │  │
│ │热切换   │    │ 性能调优   │  │
│ └─────────────┘    └─────────────┘  │
│                          │                   │         │
│                          └─────────┬─────────┘         │
│                                    ▼                   │
│                          ┌─────────────────────┐  │
│                          │  预测结果应用    │  │
│                          ├─────────────────────┤  │
│                          │ 风险等级分类    │  │
│                          │ 干预建议生成    │  │
│                          │ 趋势预测分析    │  │
│                          │ 可视化数据生成  │  │
│                          └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 8.5.3 监控与日志
- **应用监控**：
  - 系统性能监控（CPU、内存、磁盘）
  - 应用响应时间监控
  - 错误率监控
- **日志管理**：
  - 应用日志（访问日志、错误日志）
  - 数据库操作日志
  - 安全审计日志

### 8.6 安全架构
- **认证安全**：
  - JWT Token认证
  - 会话管理
  - 密码策略强制
- **数据安全**：
  - 数据传输加密（HTTPS）
  - 敏感数据存储加密
  - 数据脱敏处理
- **应用安全**：
  - SQL注入防护
  - XSS攻击防护
  - CSRF防护
  - 输入验证和过滤

## 9. 技术风险评估与应对策略

### 8.1 AI模块技术风险
- **风险描述**：聊天机器人回复质量、情感分析准确性可能影响用户体验
- **应对策略**：
  - 采用基于规则的简化实现，降低技术复杂度
  - 预先构建完善的知识库，确保回答质量
  - 实现用户满意度评价机制，持续优化回复质量
  - 情感分析采用成熟的情感词典方法，保证基础准确性

### 9.2 实时通信风险
- **风险描述**：WebSocket连接稳定性、并发性能可能影响系统可用性
- **应对策略**：
  - 实现连接重试机制和心跳检测
  - 合理设置连接超时和并发限制
  - 准备轮询方案作为备选，确保功能可用性

### 9.3 性能风险
- **风险描述**：情感分析计算开销可能影响系统响应速度
- **应对策略**：
  - 采用轻量级算法，避免复杂模型
  - 实现结果缓存机制
  - 异步处理非关键分析任务

### 9.4 数据安全与隐私风险
- **风险描述**：心理健康数据属于敏感信息，存在泄露风险
- **应对策略**：
  - 严格的数据访问权限控制
  - 敏感数据加密存储和传输
  - 数据脱敏处理，保护学生隐私
  - 定期安全审计和漏洞扫描

### 9.5 用户体验风险
- **风险描述**：AI功能可能存在理解错误、回复不当等问题
- **应对策略**：
  - 建立完善的知识库审核机制
  - 实现用户反馈和满意度评价
  - 设置人工干预和紧急联系机制
  - 持续优化和更新AI模型

## 10. 系统测试计划

### 10.1 测试策略
- **单元测试**：对每个功能模块进行独立测试，覆盖核心业务逻辑
- **集成测试**：测试模块间的接口和数据流转
- **系统测试**：端到端功能测试，验证完整的业务流程
- **AI模块测试**：重点测试聊天机器人回复质量和情感分析准确性
- **性能测试**：并发用户测试，验证系统响应时间和稳定性
- **安全测试**：SQL注入、XSS攻击、数据泄露等安全漏洞测试

### 10.2 测试用例设计

#### 10.2.1 心理测试功能测试用例
| test_case_id | test_description | input_data | expected_result | priority |
|------------|----------|----------|----------|--------|
| TC001 | 学生正常完成SCL-90测试 | 90道题目全部作答 | 生成完整测试报告和得分分析 | 高 |
| TC002 | 学生中途退出测试 | 答题30道后关闭页面 | 保存答题进度，下次可继续 | 中 |
| TC003 | 测试提交时间异常 | 10秒内完成全部题目 | 标记为异常测试，管理员可见 | 高 |
| TC004 | 预警规则触发 | SDS标准分≥63分 | 生成抑郁预警，通知管理员 | 高 |

#### 10.2.2 AI聊天机器人测试用例
| 测试用例ID | 测试描述 | 输入数据 | 预期结果 | 优先级 |
|------------|----------|----------|----------|--------|
| AI001 | 焦虑相关问题回答 | "我最近很焦虑，睡不着" | 识别焦虑情绪，提供应对建议 | 高 |
| AI002 | 紧急情况识别 | "我不想活了" | 立即提供紧急联系方式 | 高 |
| AI003 | 情感分析准确性 | "今天心情很好，通过了考试" | 情感分析结果为"正面" | 中 |
| AI004 | 多轮对话连续性 | 连续询问相关问题 | 保持对话上下文连贯性 | 中 |

#### 10.2.3 数据可视化测试用例
| 测试用例ID | 测试描述 | 测试数据 | 预期结果 | 优先级 |
|------------|----------|----------|----------|--------|
| VIS001 | 班级热力图渲染 | 50名学生的心理状态数据 | 正确显示颜色分布和交互 | 高 |
| VIS002 | 测试结果雷达图 | SCL-90各因子得分 | 正确显示10个维度的对比 | 中 |
| VIS003 | 趋势预测图展示 | 历史测试数据+预测结果 | 显示历史趋势和预测曲线 | 中 |

### 10.3 测试环境搭建
- **硬件环境**：CPU 4核、内存8GB、硬盘100GB
- **软件环境**：Python 3.9、MySQL 8.0、Redis 7.0
- **测试数据**：生成1000条模拟学生数据，包含各种测试场景
- **测试工具**：PyTest（单元测试）、Selenium（UI自动化）、JMeter（性能测试）

### 10.4 验收标准
- **功能完整性**：所有需求文档描述的功能100%实现
- **性能指标**：页面加载时间≤3秒，并发支持≥100用户
- **AI功能**：聊天机器人回复满意度≥80%，情感分析准确率≥75%
- **系统稳定性**：连续运行72小时无重大故障
- **数据安全**：通过安全渗透测试，无高危漏洞

## 11. 项目交付物清单

### 11.1 代码交付物
- **源代码**：完整的前后端源代码，包含详细注释
- **配置文件**：数据库配置、环境配置、部署配置
- **依赖文件**：requirements.txt、package.json等依赖管理文件
- **SQL脚本**：数据库建表脚本、初始化数据脚本

### 11.2 文档交付物
- **需求文档**：本需求文档的最终版本
- **设计文档**：系统架构设计、数据库设计、API设计文档
- **技术文档**：AI模块技术实现、算法说明、部署手册
- **测试文档**：测试计划、测试用例、测试报告
- **用户手册**：系统使用说明、管理员操作指南
- **论文文档**：毕业设计论文（按学校格式要求）

### 11.3 部署交付物
- **部署包**：Docker镜像或部署脚本
- **演示环境**：可在线访问的演示系统地址
- **数据备份**：测试数据和演示数据的备份文件

