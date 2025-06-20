// contracts/FirewallRules.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FirewallRules {
    // ====== 端口封锁 ======
    uint[] public blockedPorts;
    mapping(uint => bool) public isBlocked;

    // ====== 攻击模式上链 ======
    uint[] public attackPatterns;
    event AttackPatternAdded(uint pattern);

    // ====== 检测阈值 ======
    uint public detectionThreshold = 10;
    event ThresholdChanged(uint newThreshold);

    // ====== 多签名机制 ======
    address[] public signers;
    mapping(address => bool) public isSigner;
    uint public minSignatures = 2; // 默认至少2人同意
    // 端口 => 签名人 => 是否已签名
    mapping(uint => mapping(address => bool)) public portVotes;
    // 端口 => 当前已签名数
    mapping(uint => uint) public portVoteCount;
    event PortVote(address signer, uint port, uint voteCount);
    event PortBlocked(uint port);
    event SignersChanged(address[] newSigners, uint minSignatures);

    // ====== 端口管理 ======
    function addPort(uint port) public onlySigner {
        require(!isBlocked[port], "Port already blocked");
        require(!portVotes[port][msg.sender], "Already voted");
        portVotes[port][msg.sender] = true;
        portVoteCount[port] += 1;
        emit PortVote(msg.sender, port, portVoteCount[port]);
        if (portVoteCount[port] >= minSignatures) {
            blockedPorts.push(port);
            isBlocked[port] = true;
            emit PortBlocked(port);
        }
    }

    function getPort(uint index) public view returns (uint) {
        return blockedPorts[index];
    }

    function getLength() public view returns (uint) {
        return blockedPorts.length;
    }

    // ====== 攻击模式管理 ======
    function addAttackPattern(uint pattern) public onlySigner {
        attackPatterns.push(pattern);
        emit AttackPatternAdded(pattern);
    }

    function getAttackPattern(uint index) public view returns (uint) {
        return attackPatterns[index];
    }

    function getAttackPatternLength() public view returns (uint) {
        return attackPatterns.length;
    }

    // ====== 阈值设置 ======
    function setDetectionThreshold(uint threshold) public onlySigner {
        detectionThreshold = threshold;
        emit ThresholdChanged(threshold);
    }

    // ====== 自动封锁（AI检测调用） ======
    function autoBlock(uint port) public onlySigner {
        // 直接封锁，无需多签
        if (!isBlocked[port]) {
            blockedPorts.push(port);
            isBlocked[port] = true;
            emit PortBlocked(port);
        }
    }

    // ====== 多签名人管理 ======
    function setSigners(address[] memory _signers, uint _minSignatures) public onlyOwner {
        require(_signers.length > 0, "No signers");
        require(_minSignatures > 0 && _minSignatures <= _signers.length, "Invalid minSignatures");
        // 清空旧签名人
        for (uint i = 0; i < signers.length; i++) {
            isSigner[signers[i]] = false;
        }
        signers = _signers;
        minSignatures = _minSignatures;
        for (uint i = 0; i < _signers.length; i++) {
            isSigner[_signers[i]] = true;
        }
        emit SignersChanged(_signers, _minSignatures);
    }

    // ====== 权限修饰符 ======
    address public owner;
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    modifier onlySigner() {
        require(isSigner[msg.sender], "Not signer");
        _;
    }

    // ====== 构造函数 ======
    constructor() {
        owner = msg.sender;
        // 默认部署者为唯一签名人
        signers.push(owner);
        isSigner[owner] = true;
        minSignatures = 1;
    }
}
